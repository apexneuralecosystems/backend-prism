"""
Salesforce OAuth credential storage and data push (org-only).

Credentials stored per org email in data/social_credentials/{org_email}_salesforce.json.

Object creation  → SOAP Metadata API  createMetadata  (auto-deploys + auto-grants FLS)
Field creation   → SOAP Metadata API  createMetadata  (auto-deploys + auto-grants FLS)
Field check      → FieldDefinition Tooling API query  (reliable even before describe refreshes)
Record push      → Direct REST API POST               (retry-and-remove pattern from reference)

Why SOAP Metadata API for fields (not Tooling API):
  The Tooling API creates fields without auto-granting Field-Level Security.
  SOAP Metadata API (same as jsforce conn.metadata.create) deploys the field AND
  grants FLS to the System Administrator profile automatically.
  Reference: PRISM-ApexNeural/reference/CF/server.js
"""
import re
import time
import json
import requests
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

_backend_dir = Path(__file__).resolve().parent.parent
_credentials_dir = _backend_dir / "data" / "social_credentials"

SALESFORCE_AUTH_URL = "https://login.salesforce.com/services/oauth2/authorize"
SALESFORCE_TOKEN_URL = "https://login.salesforce.com/services/oauth2/token"

# Fallback API version — overridden dynamically from the org at runtime
SF_API_VERSION_FALLBACK = "59.0"

PRISM_OBJECT_API_NAME = "PRISM_Jobs__c"
PRISM_OBJECT_LABEL = "PRISM Job"
PRISM_OBJECT_PLURAL_LABEL = "PRISM Jobs"
PRISM_FIELD_API_NAME = "Job_Data__c"
PRISM_FIELD_LABEL = "Job Data"


# ──────────────────────────────────────────────
# Credential helpers
# ──────────────────────────────────────────────

def _profile_to_filename(profile_id: str) -> str:
    safe = re.sub(r"[^\w\-.]", "_", profile_id)
    safe = safe.replace(".", "_").replace("@", "_at_")
    return f"{safe}_salesforce.json"


def _ensure_cred_dir() -> Path:
    _credentials_dir.mkdir(parents=True, exist_ok=True)
    return _credentials_dir


def save_salesforce_credential(profile_id: str, data: Dict[str, Any]) -> None:
    """Persist Salesforce OAuth tokens for an org (keyed by org email)."""
    _ensure_cred_dir()
    path = _credentials_dir / _profile_to_filename(profile_id)
    payload = {
        "access_token": data.get("access_token"),
        "refresh_token": data.get("refresh_token"),
        "instance_url": data.get("instance_url"),
        "token_type": data.get("token_type", "Bearer"),
        "issued_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[SF] ✅ Credentials saved for {profile_id}")


def get_salesforce_credential(profile_id: str) -> Optional[Dict[str, Any]]:
    path = _credentials_dir / _profile_to_filename(profile_id)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def delete_salesforce_credential(profile_id: str) -> None:
    path = _credentials_dir / _profile_to_filename(profile_id)
    if path.exists():
        path.unlink()


def refresh_salesforce_token(profile_id: str, client_id: str, client_secret: str) -> Optional[str]:
    """Exchange stored refresh_token for a new access_token. Returns new token or None."""
    cred = get_salesforce_credential(profile_id)
    if not cred or not cred.get("refresh_token"):
        print(f"[SF] ⚠️  No refresh token for {profile_id}")
        return None

    resp = requests.post(
        SALESFORCE_TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": cred["refresh_token"],
        },
        timeout=15,
    )
    if resp.status_code != 200:
        print(f"[SF] ❌ Token refresh failed: {resp.status_code} {resp.text[:200]}")
        return None

    token_data = resp.json()
    new_token = token_data.get("access_token")
    if not new_token:
        return None

    cred["access_token"] = new_token
    if token_data.get("instance_url"):
        cred["instance_url"] = token_data["instance_url"]
    cred["issued_at"] = datetime.utcnow().isoformat() + "Z"
    path = _credentials_dir / _profile_to_filename(profile_id)
    with open(path, "w") as f:
        json.dump(cred, f, indent=2)
    print(f"[SF] 🔄 Access token refreshed for {profile_id}")
    return new_token


# ──────────────────────────────────────────────
# API version detection
# ──────────────────────────────────────────────

def _get_api_version(instance_url: str, access_token: str) -> str:
    """Query the org for its latest available REST API version."""
    try:
        resp = requests.get(
            f"{instance_url.rstrip('/')}/services/data/",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
        if resp.status_code == 200:
            versions = resp.json()
            if versions:
                latest = versions[-1]["version"]
                print(f"[SF] 📋 Org API version: v{latest}")
                return latest
    except Exception as e:
        print(f"[SF] ⚠️  Could not detect API version, using fallback {SF_API_VERSION_FALLBACK}: {e}")
    return SF_API_VERSION_FALLBACK


# ──────────────────────────────────────────────
# Existence checks
# ──────────────────────────────────────────────

def _object_exists_rest(instance_url: str, access_token: str, api_version: str) -> bool:
    """Return True if PRISM_Jobs__c is already deployed in this org (HTTP 200 on describe)."""
    url = f"{instance_url.rstrip('/')}/services/data/v{api_version}/sobjects/{PRISM_OBJECT_API_NAME}/describe"
    try:
        resp = requests.get(url, headers={"Authorization": f"Bearer {access_token}"}, timeout=15)
        exists = resp.status_code == 200
        print(f"[SF] 🔍 Object {PRISM_OBJECT_API_NAME} exists: {exists} (HTTP {resp.status_code})")
        return exists
    except Exception as e:
        print(f"[SF] ⚠️  Object existence check failed: {e}")
        return False


def _field_exists_via_tooling(instance_url: str, access_token: str, api_version: str) -> bool:
    """
    Check if Job_Data__c field exists using the Tooling API FieldDefinition query.
    This is the approach from reference/CF/server.js (getFieldDefinitionsFromTooling).
    Much more reliable than describe() because it queries the raw metadata store and
    returns results even when describe is stale or FLS is missing.
    """
    from urllib.parse import quote
    soql = (
        f"SELECT QualifiedApiName FROM FieldDefinition "
        f"WHERE EntityDefinition.QualifiedApiName='{PRISM_OBJECT_API_NAME}' "
        f"AND QualifiedApiName='{PRISM_FIELD_API_NAME}'"
    )
    url = f"{instance_url.rstrip('/')}/services/data/v{api_version}/tooling/query?q={quote(soql)}"
    try:
        resp = requests.get(url, headers={"Authorization": f"Bearer {access_token}"}, timeout=15)
        print(f"[SF] 🔍 FieldDefinition check (HTTP {resp.status_code}): {resp.text[:300]}")
        if resp.status_code == 200:
            return len(resp.json().get("records", [])) > 0
    except Exception as e:
        print(f"[SF] ⚠️  FieldDefinition check exception: {e}")
    return False


def _field_visible_in_describe(instance_url: str, access_token: str, api_version: str) -> bool:
    """
    Return True if Job_Data__c appears in the REST describe endpoint.
    If a field exists in FieldDefinition but NOT here, FLS has not been granted yet.
    """
    url = f"{instance_url.rstrip('/')}/services/data/v{api_version}/sobjects/{PRISM_OBJECT_API_NAME}/describe"
    try:
        resp = requests.get(url, headers={"Authorization": f"Bearer {access_token}"}, timeout=15)
        if resp.status_code != 200:
            return False
        fields = [f["name"] for f in resp.json().get("fields", [])]
        visible = PRISM_FIELD_API_NAME in fields
        print(f"[SF] 🔍 Field visible in describe: {visible} | fields: {fields}")
        return visible
    except Exception as e:
        print(f"[SF] ⚠️  Describe check exception: {e}")
        return False


# ──────────────────────────────────────────────
# SOAP Metadata API — object and field CRUD
# Using SOAP (not Tooling API) because SOAP Metadata API automatically
# deploys the metadata AND grants FLS to System Administrator.
# Reference: jsforce conn.metadata.create() in reference/CF/server.js
# ──────────────────────────────────────────────

def _soap_metadata_call(
    instance_url: str,
    access_token: str,
    api_version: str,
    operation: str,          # "createMetadata" or "deleteMetadata"
    metadata_xml: str,
    label: str,
) -> Dict[str, Any]:
    """Generic SOAP Metadata API caller used by create/delete operations."""
    import re as _re
    endpoint = f"{instance_url.rstrip('/')}/services/Soap/m/{api_version}"

    if operation == "deleteMetadata":
        # deleteMetadata takes <type> + <fullNames>, not inline metadata element
        body_inner = metadata_xml  # caller passes the delete body directly
    else:
        body_inner = metadata_xml

    envelope = f"""<?xml version="1.0" encoding="utf-8"?>
<soapenv:Envelope
    xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/"
    xmlns:met="http://soap.sforce.com/2006/04/metadata">
  <soapenv:Header>
    <met:CallOptions/>
    <met:SessionHeader><met:sessionId>{access_token}</met:sessionId></met:SessionHeader>
  </soapenv:Header>
  <soapenv:Body>
    <met:{operation}>
      {body_inner}
    </met:{operation}>
  </soapenv:Body>
</soapenv:Envelope>"""

    print(f"[SF] 📤 SOAP {operation} → {endpoint} ({label})")
    try:
        resp = requests.post(
            endpoint,
            data=envelope.encode("utf-8"),
            headers={"Content-Type": "text/xml; charset=utf-8", "SOAPAction": '""'},
            timeout=40,
        )
        body = resp.text
        print(f"[SF] 📥 SOAP {operation} response (HTTP {resp.status_code}): {body[:500]}")

        if "<success>true</success>" in body:
            print(f"[SF] ✅ SOAP {operation} succeeded for {label}")
            return {"success": True}

        lower = body.lower()
        if "already exists" in lower or "duplicate" in lower:
            print(f"[SF] ℹ️  {label} already exists — treating as success")
            return {"success": True, "existed": True}

        fault = _re.search(r"<faultstring>(.*?)</faultstring>", body, _re.DOTALL)
        err_tag = _re.search(r"<message>(.*?)</message>", body, _re.DOTALL)
        msg = (fault.group(1) if fault else None) or (err_tag.group(1) if err_tag else body[:400])
        print(f"[SF] ❌ SOAP {operation} failed for {label}: {msg.strip()}")
        return {"success": False, "error": msg.strip()}
    except Exception as exc:
        print(f"[SF] ❌ SOAP {operation} exception for {label}: {exc}")
        return {"success": False, "error": str(exc)}


def _soap_create_object(instance_url: str, access_token: str, api_version: str) -> Dict[str, Any]:
    """
    Create PRISM_Jobs__c custom object via SOAP Metadata API.
    deploymentStatus=Deployed means the object is immediately usable.
    """
    metadata_xml = f"""<met:metadata xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:type="met:CustomObject">
        <met:fullName>{PRISM_OBJECT_API_NAME}</met:fullName>
        <met:label>{PRISM_OBJECT_LABEL}</met:label>
        <met:pluralLabel>{PRISM_OBJECT_PLURAL_LABEL}</met:pluralLabel>
        <met:nameField>
          <met:type>Text</met:type>
          <met:label>Name</met:label>
        </met:nameField>
        <met:deploymentStatus>Deployed</met:deploymentStatus>
        <met:sharingModel>ReadWrite</met:sharingModel>
      </met:metadata>"""
    return _soap_metadata_call(instance_url, access_token, api_version, "createMetadata", metadata_xml, PRISM_OBJECT_API_NAME)


def _soap_create_field(instance_url: str, access_token: str, api_version: str) -> Dict[str, Any]:
    """
    Create Job_Data__c LongTextArea field on PRISM_Jobs__c via SOAP Metadata API.

    SOAP Metadata API (same as jsforce conn.metadata.create("CustomField", ...)) automatically
    deploys the field AND grants FLS to System Administrator — unlike the Tooling API which
    creates the field without any FLS grants (causing INVALID_FIELD errors forever).

    Note: Salesforce has no native JSON field type. LongTextArea (up to 131,072 chars)
    is the correct type for storing JSON payloads — this is the standard Salesforce approach.
    """
    metadata_xml = f"""<met:metadata xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:type="met:CustomField">
        <met:fullName>{PRISM_OBJECT_API_NAME}.{PRISM_FIELD_API_NAME}</met:fullName>
        <met:label>{PRISM_FIELD_LABEL}</met:label>
        <met:type>LongTextArea</met:type>
        <met:length>131072</met:length>
        <met:visibleLines>10</met:visibleLines>
      </met:metadata>"""
    return _soap_metadata_call(instance_url, access_token, api_version, "createMetadata", metadata_xml, PRISM_FIELD_API_NAME)


def _soap_delete_field(instance_url: str, access_token: str, api_version: str) -> Dict[str, Any]:
    """
    Delete the existing broken Job_Data__c field via SOAP Metadata API deleteMetadata.
    Used when FieldDefinition confirms the field exists but it has no FLS (created by
    the old Tooling API code without FLS grants).  After deletion we recreate via SOAP
    which auto-grants FLS.
    """
    delete_body = f"""<met:type>CustomField</met:type>
      <met:fullNames>{PRISM_OBJECT_API_NAME}.{PRISM_FIELD_API_NAME}</met:fullNames>"""
    return _soap_metadata_call(instance_url, access_token, api_version, "deleteMetadata", delete_body, f"delete {PRISM_FIELD_API_NAME}")


def _grant_fls_for_connected_user(instance_url: str, access_token: str, api_version: str) -> None:
    """
    Grant ObjectPermissions + FieldPermissions for the current OAuth user's profile.

    Why this is needed:
      SOAP Metadata API createMetadata creates the field schema but does NOT
      auto-grant Field-Level Security in org-farm / Developer Edition orgs.
      Without FLS the field is invisible in describe() and DML raises INVALID_FIELD.

    Approach (Data API, not Tooling API):
      1. GET /services/oauth2/userinfo          → current user_id
      2. GET /sobjects/User/{user_id}           → ProfileId
      3. SOQL PermissionSet WHERE ProfileId     → profile-owned PermissionSet Id
      4. POST /sobjects/ObjectPermissions       → grant CRUD on PRISM_Jobs__c
      5. POST /sobjects/FieldPermissions        → grant read+edit on Job_Data__c
    """
    from urllib.parse import quote
    api_base = f"{instance_url.rstrip('/')}/services/data/v{api_version}"
    hdrs = {"Authorization": f"Bearer {access_token}"}
    hdrs_json = {**hdrs, "Content-Type": "application/json"}

    print(f"[SF] 🔐 Granting FLS for connected user ...")

    # ── 1. Get current user_id ───────────────────────────────
    try:
        ui = requests.get(f"{instance_url.rstrip('/')}/services/oauth2/userinfo", headers=hdrs, timeout=10)
        print(f"[SF] 🔐 userinfo HTTP {ui.status_code}: {ui.text[:200]}")
        user_id = ui.json().get("user_id") if ui.status_code == 200 else None
    except Exception as e:
        print(f"[SF] ⚠️  userinfo exception: {e}")
        user_id = None

    if not user_id:
        print(f"[SF] ⚠️  Cannot grant FLS — user_id not found")
        return

    # ── 2. Get ProfileId from User record ───────────────────
    try:
        u = requests.get(f"{api_base}/sobjects/User/{user_id}?fields=ProfileId,Name", headers=hdrs, timeout=10)
        print(f"[SF] 🔐 User HTTP {u.status_code}: {u.text[:200]}")
        profile_id = u.json().get("ProfileId") if u.status_code == 200 else None
        print(f"[SF] 🔐 User: {u.json().get('Name','?')} | ProfileId: {profile_id}")
    except Exception as e:
        print(f"[SF] ⚠️  User query exception: {e}")
        profile_id = None

    if not profile_id:
        print(f"[SF] ⚠️  Cannot grant FLS — ProfileId not found")
        return

    # ── 3. Get profile-owned PermissionSet ──────────────────
    soql = f"SELECT Id, Name FROM PermissionSet WHERE IsOwnedByProfile=true AND ProfileId='{profile_id}'"
    try:
        ps = requests.get(f"{api_base}/query/?q={quote(soql)}", headers=hdrs, timeout=10)
        print(f"[SF] 🔐 PermissionSet HTTP {ps.status_code}: {ps.text[:300]}")
        records = ps.json().get("records", []) if ps.status_code == 200 else []
        ps_id = records[0]["Id"] if records else None
        ps_name = records[0]["Name"] if records else "?"
        print(f"[SF] 🔐 PermissionSet: '{ps_name}' | ID: {ps_id}")
    except Exception as e:
        print(f"[SF] ⚠️  PermissionSet query exception: {e}")
        ps_id = None

    if not ps_id:
        print(f"[SF] ⚠️  Cannot grant FLS — PermissionSet not found for profile {profile_id}")
        return

    # ── 4. Grant ObjectPermissions for PRISM_Jobs__c ────────
    try:
        op = requests.post(
            f"{api_base}/sobjects/ObjectPermissions",
            json={
                "ParentId": ps_id,
                "SobjectType": PRISM_OBJECT_API_NAME,
                "PermissionsRead": True,
                "PermissionsCreate": True,
                "PermissionsEdit": True,
                "PermissionsDelete": True,
                "PermissionsViewAllRecords": True,
                "PermissionsModifyAllRecords": True,
            },
            headers=hdrs_json, timeout=15,
        )
        ob = op.text[:200]
        if op.status_code in (200, 201):
            print(f"[SF] ✅ ObjectPermissions granted for {PRISM_OBJECT_API_NAME}")
        elif "DUPLICATE_VALUE" in ob or "already" in ob.lower() or "FIELD_INTEGRITY_EXCEPTION" in ob:
            print(f"[SF] ℹ️  ObjectPermissions already set for {PRISM_OBJECT_API_NAME}")
        else:
            print(f"[SF] ⚠️  ObjectPermissions HTTP {op.status_code}: {ob}")
    except Exception as e:
        print(f"[SF] ⚠️  ObjectPermissions exception: {e}")

    # ── 5. Grant FieldPermissions for Job_Data__c ───────────
    # IMPORTANT: "SobjectType" is a required restricted picklist field.
    # Omitting it (or sending null) causes:
    #   INVALID_OR_NULL_FOR_RESTRICTED_PICKLIST: "Sobject Type Name: bad value for restricted picklist field: null"
    try:
        fp = requests.post(
            f"{api_base}/sobjects/FieldPermissions",
            json={
                "ParentId": ps_id,
                "SobjectType": PRISM_OBJECT_API_NAME,               # required — DO NOT omit
                "Field": f"{PRISM_OBJECT_API_NAME}.{PRISM_FIELD_API_NAME}",
                "PermissionsRead": True,
                "PermissionsEdit": True,
            },
            headers=hdrs_json, timeout=15,
        )
        fb = fp.text[:200]
        if fp.status_code in (200, 201):
            print(f"[SF] ✅ FieldPermissions granted for {PRISM_FIELD_API_NAME}")
        elif "DUPLICATE_VALUE" in fb or "already" in fb.lower():
            print(f"[SF] ℹ️  FieldPermissions already set for {PRISM_FIELD_API_NAME}")
        else:
            print(f"[SF] ⚠️  FieldPermissions HTTP {fp.status_code}: {fb}")
    except Exception as e:
        print(f"[SF] ⚠️  FieldPermissions exception: {e}")


def _ensure_field_ready(instance_url: str, access_token: str, api_version: str) -> Dict[str, Any]:
    """
    Ensure Job_Data__c field exists AND is accessible (has FLS).

    Root cause in org-farm Developer Edition orgs:
      SOAP Metadata API creates the field schema (confirmed by FieldDefinition query)
      but does NOT auto-grant Field-Level Security. Without FLS the field never appears
      in describe() and polling for 90 s is pointless — we must grant FLS explicitly.

    Flow:
      A. Field visible in describe → already has FLS, nothing to do.
      B. Field not in describe, not in FieldDefinition → create via SOAP + grant FLS.
      C. Field in FieldDefinition but NOT in describe → ghost/no-FLS → delete + recreate + grant FLS.
      After FLS grant: wait 10 s, check describe once, proceed regardless.
    """
    # ── A: already fully accessible ──────────────────────────
    if _field_visible_in_describe(instance_url, access_token, api_version):
        print(f"[SF] ✅ Field {PRISM_FIELD_API_NAME} already accessible — skipping setup")
        return {"success": True}

    # ── B/C: field not visible, check raw metadata ───────────
    field_in_metadata = _field_exists_via_tooling(instance_url, access_token, api_version)

    if field_in_metadata:
        # Exists in schema but no FLS → delete via SOAP and recreate fresh
        print(f"[SF] ⚠️  Field in metadata but not in describe → no FLS. Deleting and recreating ...")
        del_result = _soap_delete_field(instance_url, access_token, api_version)
        if del_result["success"]:
            print(f"[SF] ⏳ Waiting 5 s after deletion ...")
            time.sleep(5)
        else:
            print(f"[SF] ⚠️  Delete result: {del_result.get('error')} — proceeding ...")

    # ── Create fresh field via SOAP ───────────────────────────
    print(f"[SF] 🏗️  Creating {PRISM_FIELD_API_NAME} via SOAP Metadata API ...")
    create_result = _soap_create_field(instance_url, access_token, api_version)
    if not create_result["success"]:
        return {"success": False, "error": f"Field creation failed: {create_result['error']}"}

    # ── Grant FLS explicitly (SOAP doesn't auto-grant FLS in org-farm orgs) ──
    print(f"[SF] ⏳ Waiting 5 s for schema to register before granting FLS ...")
    time.sleep(5)
    _grant_fls_for_connected_user(instance_url, access_token, api_version)

    # ── Wait for FLS propagation and verify ──────────────────
    print(f"[SF] ⏳ Waiting 10 s for FLS to propagate ...")
    time.sleep(10)

    if _field_visible_in_describe(instance_url, access_token, api_version):
        print(f"[SF] ✅ Field {PRISM_FIELD_API_NAME} now visible in describe after FLS grant!")
        return {"success": True}

    # FLS might need a bit more time — check one more time after extra wait
    print(f"[SF] ⏳ FLS not visible yet — waiting another 10 s ...")
    time.sleep(10)
    if _field_visible_in_describe(instance_url, access_token, api_version):
        print(f"[SF] ✅ Field {PRISM_FIELD_API_NAME} visible after extended FLS wait!")
        return {"success": True}

    # Proceed anyway — push retry loop will handle residual propagation lag
    print(f"[SF] ⚠️  Field not yet in describe after FLS grant — proceeding to push (retry loop will wait for Job_Data__c)")
    return {"success": True}


# ──────────────────────────────────────────────
# Record push — direct REST API with retry-and-remove
# Pattern from reference/CF/server.js POST /api/objects/:obj/records
# ──────────────────────────────────────────────

def _parse_no_such_column(msg: str) -> Optional[str]:
    """Extract field name from Salesforce 'No such column X' error message."""
    import re as _re
    m = _re.search(r"No such column '([^']+)'", msg, _re.IGNORECASE)
    return m.group(1) if m else None


def _push_record_direct(
    instance_url: str,
    access_token: str,
    api_version: str,
    record_name: str,
    job_data: Dict[str, Any],
    max_retries: int = 8,
) -> Dict[str, Any]:
    """
    Create a PRISM_Jobs__c record via direct REST API POST.

    Retry rules (IMPORTANT — prevents silent data loss):
    - If 'No such column Job_Data__c' → the field is still propagating.
      WAIT 12 s and retry with FULL body (never drop Job_Data__c — doing so
      creates a name-only record with no job data, which is useless).
    - If 'No such column OTHER_FIELD' → remove that unknown field and retry
      (follows reference/CF/server.js pattern for other fields).
    - Any other error → fail immediately with message.
    """
    url = f"{instance_url.rstrip('/')}/services/data/v{api_version}/sobjects/{PRISM_OBJECT_API_NAME}"
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}

    # Compact JSON; stay well within the 131,072-char LongTextArea limit
    job_data_str = json.dumps(job_data, ensure_ascii=False, separators=(",", ":"), default=str)
    if len(job_data_str) > 130000:
        print(f"[SF] ⚠️  job_data JSON is {len(job_data_str)} chars — truncating to 130000")
        job_data_str = job_data_str[:130000] + "...}"
    print(f"[SF] 📊 Job data size: {len(job_data_str)} chars")

    body: Dict[str, Any] = {
        "Name": record_name[:255],
        PRISM_FIELD_API_NAME: job_data_str,
    }
    print(f"[SF] 📤 Pushing record '{record_name}' → {url}")

    last_error = "Unknown error"
    for attempt in range(max_retries + 1):
        print(f"[SF] 🔁 Push attempt {attempt + 1}/{max_retries + 1} | body fields: {list(body.keys())}")
        try:
            resp = requests.post(url, json=body, headers=headers, timeout=30)
            data = resp.json() if resp.text else {}

            if resp.status_code in (200, 201) and isinstance(data, dict) and data.get("success"):
                record_id = data.get("id", "")
                has_data = PRISM_FIELD_API_NAME in body
                print(f"[SF] ✅ Record created! ID: {record_id} | Job_Data included: {has_data}")
                return {"success": True, "record_id": record_id}

            # Parse Salesforce error
            if isinstance(data, list) and data:
                msg = data[0].get("message", str(data[0]))
                error_code = data[0].get("errorCode", "")
            elif isinstance(data, dict):
                msg = data.get("message", str(data))
                error_code = data.get("errorCode", "")
            else:
                msg = resp.text[:400]
                error_code = ""

            last_error = msg
            print(f"[SF] ⚠️  Attempt {attempt + 1} failed (HTTP {resp.status_code}) | {error_code}: {msg[:200]}")

            bad_field = _parse_no_such_column(msg)

            if bad_field == PRISM_FIELD_API_NAME and attempt < max_retries:
                # Our main data field isn't propagated yet — WAIT and retry WITH the data.
                # DO NOT remove it: creating a record without Job_Data__c is useless.
                wait_s = 12
                print(f"[SF] ⏳ {PRISM_FIELD_API_NAME} still propagating — waiting {wait_s} s (attempt {attempt + 1}/{max_retries}) ...")
                time.sleep(wait_s)
                continue

            if bad_field and bad_field in body and bad_field != PRISM_FIELD_API_NAME and attempt < max_retries:
                # Some other unrecognised field — safe to drop and retry
                print(f"[SF] 🗑️  Removing unknown field '{bad_field}' from body and retrying ...")
                body = {k: v for k, v in body.items() if k != bad_field}
                time.sleep(3)
                continue

            return {"success": False, "error": last_error}

        except Exception as exc:
            last_error = str(exc)
            print(f"[SF] ⚠️  Push attempt {attempt + 1} exception: {last_error[:200]}")
            if attempt < max_retries:
                time.sleep(5)
                continue
            return {"success": False, "error": last_error}

    return {
        "success": False,
        "error": (
            f"Push failed after {max_retries + 1} attempts — {PRISM_FIELD_API_NAME} "
            "is still propagating. Please try again in 1–2 minutes."
        ),
    }


# ──────────────────────────────────────────────
# Main entry points
# ──────────────────────────────────────────────

def ensure_prism_object_and_update(
    access_token: str,
    instance_url: str,
    job_data: Dict[str, Any],
    record_id: str,
) -> Dict[str, Any]:
    """
    PATCH an existing PRISM_Jobs__c record with fresh job data.
    Used when the job was already pushed (salesforce_pushed=True) to update
    the Salesforce record rather than creating a duplicate.
    Also fixes records that were created without Job_Data__c due to propagation lag.
    """
    print(f"\n[SF] ═══════════════════════════════════════")
    print(f"[SF] Updating existing record ID: {record_id}")
    print(f"[SF] ═══════════════════════════════════════")

    try:
        api_version = _get_api_version(instance_url, access_token)

        # Ensure the field is accessible before updating
        field_result = _ensure_field_ready(instance_url, access_token, api_version)
        if not field_result["success"]:
            return {"success": False, "error": f"Field not ready: {field_result['error']}"}

        url = f"{instance_url.rstrip('/')}/services/data/v{api_version}/sobjects/{PRISM_OBJECT_API_NAME}/{record_id}"
        headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}

        job_data_str = json.dumps(job_data, ensure_ascii=False, separators=(",", ":"), default=str)
        if len(job_data_str) > 130000:
            job_data_str = job_data_str[:130000] + "...}"

        print(f"[SF] 📤 PATCH {url}")
        resp = requests.patch(
            url,
            json={PRISM_FIELD_API_NAME: job_data_str},
            headers=headers,
            timeout=30,
        )
        print(f"[SF] 📥 PATCH response (HTTP {resp.status_code}): {resp.text[:200]}")

        # Salesforce PATCH returns 204 No Content on success
        if resp.status_code == 204:
            print(f"[SF] ✅ Record {record_id} updated with job data")
            return {"success": True, "record_id": record_id}

        err = resp.text[:400]
        return {"success": False, "error": f"PATCH failed HTTP {resp.status_code}: {err}"}

    except Exception as exc:
        print(f"[SF] ❌ Update exception: {exc}")
        return {"success": False, "error": str(exc)}


def ensure_prism_object_and_push(
    access_token: str,
    instance_url: str,
    job_data: Dict[str, Any],
    record_name: str,
) -> Dict[str, Any]:
    """
    Full push flow — based on reference/CF/server.js:

      1. Detect org API version dynamically.
      2. Check / create PRISM_Jobs__c object  (SOAP Metadata API — auto-grants FLS).
      3. Check / create Job_Data__c field:
           a. Field visible in describe → done.
           b. Field in metadata but not in describe (Tooling ghost, no FLS) →
              delete via SOAP + recreate via SOAP (auto-grants FLS).
           c. Field absent → create via SOAP.
      4. Push record via direct REST API with retry-and-remove pattern.
    """
    print(f"\n[SF] ═══════════════════════════════════════")
    print(f"[SF] Starting push → instance: {instance_url}")
    print(f"[SF] Record name  : {record_name}")
    print(f"[SF] ═══════════════════════════════════════")

    try:
        # ── Step 1: detect API version ──────────────────
        api_version = _get_api_version(instance_url, access_token)

        # ── Step 2: ensure PRISM_Jobs__c object exists ──
        obj_exists = _object_exists_rest(instance_url, access_token, api_version)
        if not obj_exists:
            print(f"[SF] 🏗️  Object {PRISM_OBJECT_API_NAME} not found — creating via SOAP ...")
            obj_result = _soap_create_object(instance_url, access_token, api_version)
            if not obj_result["success"]:
                return {"success": False, "error": f"Failed to create {PRISM_OBJECT_API_NAME}: {obj_result['error']}"}
            print(f"[SF] ⏳ Waiting 8 s for object to register before adding field ...")
            time.sleep(8)

        # ── Step 3: ensure Job_Data__c field exists with FLS ─
        field_result = _ensure_field_ready(instance_url, access_token, api_version)
        if not field_result["success"]:
            return {"success": False, "error": f"Field provisioning failed: {field_result['error']}"}

        # Small propagation buffer before first DML
        print(f"[SF] ⏳ Waiting 4 s before push ...")
        time.sleep(4)

        # ── Step 4: push record (retry-and-remove on propagation lag) ─
        return _push_record_direct(instance_url, access_token, api_version, record_name, job_data)

    except Exception as exc:
        print(f"[SF] ❌ Unexpected error in ensure_prism_object_and_push: {exc}")
        return {"success": False, "error": str(exc)}
