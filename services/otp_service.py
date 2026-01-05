import random
import string
from datetime import datetime, timedelta
from typing import Optional


def generate_otp() -> str:
    """Generate a 6-digit OTP"""
    return ''.join(random.choices(string.digits, k=6))


def get_otp_expiry() -> datetime:
    """Get OTP expiry time (10 minutes from now)"""
    return datetime.utcnow() + timedelta(minutes=10)


async def store_otp(db, email: str, otp: str):
    """Store OTP in MongoDB with TTL"""
    otp_collection = db["otps"]
    
    # Delete existing OTP for this email
    await otp_collection.delete_one({"email": email})
    
    # Create new OTP document
    otp_doc = {
        "email": email,
        "otp": otp,
        "attempts": 0,
        "created_at": datetime.utcnow(),
        "expires_at": get_otp_expiry()
    }
    
    result = await otp_collection.insert_one(otp_doc)
    
    # Create TTL index on expires_at field if it doesn't exist (only once)
    try:
        # Check if index already exists
        indexes = await otp_collection.list_indexes().to_list(length=100)
        index_names = [idx["name"] for idx in indexes]
        
        # Remove old TTL index on created_at if it exists (wrong field)
        if "created_at_1" in index_names:
            try:
                await otp_collection.drop_index("created_at_1")
                print(f"✅ Removed old TTL index on created_at")
            except Exception as e:
                print(f"⚠️ Could not remove old index: {e}")
        
        # Create TTL index on expires_at field if it doesn't exist
        if "expires_at_1" not in index_names:
            # Create TTL index on expires_at field (expireAfterSeconds=0 means use the date in expires_at)
            await otp_collection.create_index("expires_at", expireAfterSeconds=0)
            print(f"✅ Created TTL index on expires_at for OTP collection")
    except Exception as e:
        print(f"⚠️ Index creation warning (may already exist): {e}")
    
    print(f"✅ OTP stored for {email}: {otp} (expires at {otp_doc['expires_at']})")
    return True


async def verify_otp(db, email: str, otp: str) -> dict:
    """Verify OTP and handle attempts"""
    otp_collection = db["otps"]
    
    print(f"🔍 Verifying OTP for {email}, OTP provided: {otp}")
    
    otp_doc = await otp_collection.find_one({"email": email})
    
    if not otp_doc:
        print(f"❌ OTP not found in database for {email}")
        # Debug: Check all OTPs in collection
        all_otps = await otp_collection.find({}).to_list(length=10)
        print(f"📋 Current OTPs in database: {[{'email': doc.get('email'), 'otp': doc.get('otp'), 'expires_at': doc.get('expires_at')} for doc in all_otps]}")
        return {"success": False, "message": "OTP not found or expired"}
    
    print(f"✅ OTP document found: email={otp_doc.get('email')}, stored_otp={otp_doc.get('otp')}, expires_at={otp_doc.get('expires_at')}")
    
    # Check if expired
    if datetime.utcnow() > otp_doc["expires_at"]:
        print(f"❌ OTP expired for {email}")
        await otp_collection.delete_one({"email": email})
        return {"success": False, "message": "OTP has expired"}
    
    # Check if OTP matches
    if otp_doc["otp"] == otp:
        print(f"✅ OTP verified successfully for {email}")
        await otp_collection.delete_one({"email": email})
        return {"success": True, "message": "OTP verified successfully"}
    
    # Increment attempts
    attempts = otp_doc.get("attempts", 0) + 1
    print(f"⚠️ Invalid OTP attempt {attempts}/3 for {email}")
    
    if attempts >= 3:
        print(f"❌ Maximum attempts reached for {email}")
        await otp_collection.delete_one({"email": email})
        return {
            "success": False,
            "message": "Maximum attempts reached. Please request a new OTP.",
            "max_attempts_reached": True
        }
    
    await otp_collection.update_one(
        {"email": email},
        {"$set": {"attempts": attempts}}
    )
    
    return {
        "success": False,
        "message": f"Invalid OTP. {3 - attempts} attempts remaining.",
        "attempts_remaining": 3 - attempts
    }
