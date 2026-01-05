import requests
from datetime import datetime, timedelta
import pytz
from typing import List, Tuple, Dict, Any

def fetch_calendar_busy_periods(ics_url: str) -> List[Tuple[datetime, datetime]]:
    """
    Fetch and parse ICS calendar data to get busy periods
    
    Args:
        ics_url: URL to the ICS calendar file
        
    Returns:
        List of tuples (start_time, end_time) in IST timezone
    """
    try:
        ics_data = requests.get(ics_url, timeout=10).text
    except Exception as e:
        print(f"❌ Error fetching calendar from {ics_url}: {e}")
        return []

    events = []
    event = {}
    current_field = None

    for line in ics_data.splitlines():
        # Handle line continuations
        if line.startswith((' ', '\t')) and current_field:
            event[current_field] = event.get(current_field, '') + line.strip()
            continue

        line = line.strip()

        if line == "BEGIN:VEVENT":
            event = {}
            current_field = None
        elif line.startswith("DTSTART"):
            event["start"] = line.split(":", 1)[1] if ":" in line else ""
            current_field = "start"
        elif line.startswith("DTEND"):
            event["end"] = line.split(":", 1)[1] if ":" in line else ""
            current_field = "end"
        elif line.startswith("SUMMARY"):
            event["summary"] = line.split(":", 1)[1] if ":" in line else "Event"
            current_field = "summary"
        elif line == "END:VEVENT":
            if event and event.get("start") and event.get("end"):
                events.append(event)
            current_field = None
        else:
            current_field = None

    utc = pytz.utc
    ist = pytz.timezone("Asia/Kolkata")

    busy_periods = []
    for e in events:
        try:
            start_str = e.get("start", "")
            end_str = e.get("end", "")

            if not start_str or not end_str:
                continue

            # Handle UTC format (ends with Z)
            if start_str.endswith("Z"):
                start = datetime.strptime(start_str, "%Y%m%dT%H%M%SZ").replace(tzinfo=utc).astimezone(ist)
                end = datetime.strptime(end_str, "%Y%m%dT%H%M%SZ").replace(tzinfo=utc).astimezone(ist)
            elif "T" in start_str:
                # Local time format
                start = datetime.strptime(start_str, "%Y%m%dT%H%M%S").replace(tzinfo=ist)
                end = datetime.strptime(end_str, "%Y%m%dT%H%M%S").replace(tzinfo=ist)
            else:
                # All-day event, skip
                continue

            busy_periods.append((start, end))
        except Exception as ex:
            print(f"⚠️ Error parsing event: {ex}")
            continue

    return busy_periods


def is_slot_busy(slot_start: datetime, slot_end: datetime, busy_list: List[Tuple[datetime, datetime]]) -> bool:
    """
    Check if a time slot overlaps with any busy period
    
    Args:
        slot_start: Start time of the slot
        slot_end: End time of the slot
        busy_list: List of (start, end) tuples for busy periods
        
    Returns:
        True if slot is busy, False otherwise
    """
    for b_start, b_end in busy_list:
        # Check for overlap: slot_start < b_end and slot_end > b_start
        if slot_start < b_end and slot_end > b_start:
            return True
    return False


def get_group_free_slots(calendar_links: List[str], days: int = 5) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get free time slots for a group of people (team)
    
    Args:
        calendar_links: List of ICS calendar URLs
        days: Number of working days to check (default 5)
        
    Returns:
        Dictionary with dates as keys and list of free slots as values
        Format: {
            "Monday 30-Dec-2025": [
                {"start": "08:00 AM", "end": "08:30 AM", "slot_id": "..."},
                ...
            ]
        }
    """
    ist = pytz.timezone("Asia/Kolkata")
    
    # Fetch busy periods for each calendar
    all_busy_periods = {}
    for link in calendar_links:
        all_busy_periods[link] = fetch_calendar_busy_periods(link)
    
    # Get next N working days
    def next_working_days(n: int) -> List[datetime]:
        d = datetime.now(ist)
        days_list = []
        while len(days_list) < n:
            if d.weekday() < 5:  # Monday=0, Friday=4
                days_list.append(d.replace(hour=0, minute=0, second=0, microsecond=0))
            d += timedelta(days=1)
        return days_list
    
    workdays = next_working_days(days)
    result = {}
    now = datetime.now(ist)
    
    # Check each day
    for day in workdays:
        day_key = day.strftime("%A %d-%b-%Y")
        free_slots = []
        
        start_of_day = day.replace(hour=8, minute=0)
        end_of_day = day.replace(hour=18, minute=0)
        
        # If today, start from 5 hours from current time (rounded to next 30 min)
        if day.date() == now.date():
            # Calculate 5 hours from now
            five_hours_from_now = now + timedelta(hours=5)
            
            # Round up to next 30-minute slot
            if five_hours_from_now.minute < 30:
                start_time = five_hours_from_now.replace(minute=30, second=0, microsecond=0)
            else:
                start_time = (five_hours_from_now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
            
            # Also ensure we don't show past times - if 5 hours from now is in the past, use current time rounded up
            current_rounded = now
            if current_rounded.minute < 30:
                current_rounded = current_rounded.replace(minute=30, second=0, microsecond=0)
            else:
                current_rounded = (current_rounded + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
            
            # Use the later of: start_of_day, current_rounded, or start_time (5 hours from now)
            cursor = max(start_of_day, current_rounded, start_time)
        else:
            cursor = start_of_day
        
        while cursor < end_of_day:
            slot_end = cursor + timedelta(minutes=30)
            
            # Check if at least one interviewer is free in this slot
            # Slot is shown if ANY interviewer is available (not all busy)
            all_busy = True
            for link, busy_list in all_busy_periods.items():
                # Get busy periods for this day
                day_busy = [
                    (b_start, b_end) for (b_start, b_end) in busy_list
                    if b_start.date() == day.date() or (b_start.date() < day.date() and b_end.date() >= day.date())
                ]
                
                if not is_slot_busy(cursor, slot_end, day_busy):
                    all_busy = False
                    break
            
            # If not all busy, slot is free
            if not all_busy:
                slot_id = f"{day.strftime('%Y%m%d')}_{cursor.strftime('%H%M')}_{slot_end.strftime('%H%M')}"
                free_slots.append({
                    "start": cursor.strftime("%I:%M %p"),
                    "end": slot_end.strftime("%I:%M %p"),
                    "start_datetime": cursor.isoformat(),
                    "end_datetime": slot_end.isoformat(),
                    "slot_id": slot_id
                })
            
            cursor = slot_end
        
        # Only add days that have free slots
        if free_slots:
            result[day_key] = free_slots
    
    return result


def find_free_team_members_at_time(
    team_members: List[Dict[str, Any]], 
    selected_datetime: datetime,
    duration_minutes: int = 30
) -> List[Dict[str, Any]]:
    """
    Find team members who are free at the selected time
    
    Args:
        team_members: List of team members with calendar_link
        selected_datetime: The selected interview datetime in IST
        duration_minutes: Duration of the interview in minutes
        
    Returns:
        List of free team members
    """
    ist = pytz.timezone("Asia/Kolkata")
    slot_end = selected_datetime + timedelta(minutes=duration_minutes)
    
    free_members = []
    
    for member in team_members:
        calendar_link = member.get("calendar_link", "")
        if not calendar_link:
            continue
        
        # Fetch busy periods for this member
        busy_periods = fetch_calendar_busy_periods(calendar_link)
        
        # Check if member is free during the selected slot
        is_busy = is_slot_busy(selected_datetime, slot_end, busy_periods)
        
        if not is_busy:
            free_members.append(member)
    
    return free_members


def create_calendar_invite(
    meeting_link: str,
    start_datetime: datetime,
    end_datetime: datetime,
    summary: str,
    description: str,
    organizer_email: str,
    attendee_emails: List[str]
) -> str:
    """
    Create ICS calendar invite content
    
    Returns:
        ICS file content as string
    """
    # Convert to UTC for ICS format
    utc = pytz.utc
    start_utc = start_datetime.astimezone(utc)
    end_utc = end_datetime.astimezone(utc)
    
    start_str = start_utc.strftime("%Y%m%dT%H%M%SZ")
    end_str = end_utc.strftime("%Y%m%dT%H%M%SZ")
    
    import uuid
    uid = str(uuid.uuid4())
    
    attendees_lines = "\n".join([f"ATTENDEE:mailto:{email}" for email in attendee_emails])
    
    ics = f"""BEGIN:VCALENDAR
VERSION:2.0
CALSCALE:GREGORIAN
METHOD:REQUEST
BEGIN:VEVENT
DTSTART:{start_str}
DTEND:{end_str}
SUMMARY:{summary}
UID:{uid}
DESCRIPTION:{description}
LOCATION:{meeting_link}
ORGANIZER:mailto:{organizer_email}
{attendees_lines}
END:VEVENT
END:VCALENDAR
"""
    return ics

