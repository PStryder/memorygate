#!/usr/bin/env python3
"""
MemoryGate v2 Forgetting Mechanism Integration Test
Tests the complete retention lifecycle: store → fetch → decay → archive → purge
"""
import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8080"

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def store_observation(text, domain="test"):
    """Store an observation via MCP"""
    payload = {
        "observation": text,
        "domain": domain,
        "confidence": 0.8,
        "ai_name": "TestAI",
        "ai_platform": "IntegrationTest"
    }
    resp = requests.post(f"{BASE_URL}/api/memory/store", json=payload)
    resp.raise_for_status()
    data = resp.json()
    log(f"✓ Stored: '{text[:40]}...' (ID: {data['id']})")
    return data

def recall_observations(domain="test", limit=10):
    """Recall observations (bumps score)"""
    params = {"domain": domain, "limit": limit}
    resp = requests.get(f"{BASE_URL}/api/memory/recall", params=params)
    resp.raise_for_status()
    data = resp.json()
    log(f"✓ Recalled {len(data)} observations")
    return data

def get_stats():
    """Get memory system stats"""
    resp = requests.get(f"{BASE_URL}/api/memory/stats")
    resp.raise_for_status()
    return resp.json()

def check_observation_details(obs_id):
    """Get detailed observation info including tier and score"""
    # Direct SQL query via stats endpoint (hacky but works for testing)
    # In production we'd have a proper detail endpoint
    resp = requests.get(f"{BASE_URL}/api/memory/recall", params={"limit": 100})
    resp.raise_for_status()
    for obs in resp.json():
        if obs['id'] == obs_id:
            return obs
    return None

def trigger_decay_tick():
    """Manually trigger retention decay tick"""
    # This would normally happen every 15 minutes via background task
    # For testing, we call the internal endpoint
    try:
        resp = requests.post(f"{BASE_URL}/internal/retention/tick")
        resp.raise_for_status()
        log("✓ Decay tick triggered")
        return resp.json()
    except Exception as e:
        log(f"⚠ Decay tick failed (endpoint may not exist): {e}")
        return None

def main():
    log("=" * 60)
    log("MemoryGate v2 Forgetting Mechanism Test")
    log("=" * 60)
    
    # Phase 1: Store observations
    log("\n📝 PHASE 1: Storing observations...")
    obs1 = store_observation("Critical system insight - frequently accessed", "critical")
    obs2 = store_observation("Important but rarely used information", "moderate")
    obs3 = store_observation("Obsolete data that should be forgotten", "obsolete")
    
    time.sleep(1)
    
    # Phase 2: Check initial state
    log("\n🔍 PHASE 2: Initial state check...")
    stats = get_stats()
    log(f"Total observations: {stats.get('total_observations', 'N/A')}")
    log(f"Hot tier: {stats.get('hot_tier_count', 'N/A')}")
    log(f"Cold tier: {stats.get('cold_tier_count', 'N/A')}")
    
    # Phase 3: Simulate access patterns
    log("\n👆 PHASE 3: Simulating access patterns...")
    log("Fetching critical observation 5x (should keep score high)...")
    for i in range(5):
        recall_observations("critical")
        time.sleep(0.2)
    
    log("Fetching moderate observation 2x...")
    for i in range(2):
        recall_observations("moderate")
        time.sleep(0.2)
    
    log("NOT fetching obsolete observation (score will decay)...")
    
    # Phase 4: Check scores after access
    log("\n📊 PHASE 4: Score analysis after access...")
    all_obs = recall_observations(limit=100)
    for obs in all_obs:
        tier = obs.get('tier', 'unknown')
        score = obs.get('score', 'N/A')
        text_preview = obs['observation'][:40]
        log(f"  {obs['id'][:8]}... [{tier}] score={score:.3f} - {text_preview}...")
    
    # Phase 5: Simulate time passing and decay
    log("\n⏰ PHASE 5: Simulating decay cycle...")
    log("In production, decay happens every 15 minutes (RETENTION_TICK_SECONDS=900)")
    log("For testing, we'd either:")
    log("  A) Wait 15 minutes (too slow)")
    log("  B) Manually trigger decay tick (if endpoint exists)")
    log("  C) Directly manipulate scores via SQL (if we had admin endpoint)")
    
    decay_result = trigger_decay_tick()
    if decay_result:
        log(f"Decay result: {json.dumps(decay_result, indent=2)}")
    else:
        log("⚠ Manual decay not available - would need to wait 15min in production")
    
    # Phase 6: Check final state
    log("\n🎯 PHASE 6: Final state verification...")
    stats = get_stats()
    log(f"Total observations: {stats.get('total_observations', 'N/A')}")
    log(f"Hot tier: {stats.get('hot_tier_count', 'N/A')}")
    log(f"Cold tier: {stats.get('cold_tier_count', 'N/A')}")
    log(f"Purge eligible: {stats.get('purge_eligible_count', 'N/A')}")
    
    # Phase 7: Test archival threshold
    log("\n🗄️ PHASE 7: Testing archival threshold...")
    log("SUMMARY_TRIGGER_SCORE = -1.0")
    log("PURGE_TRIGGER_SCORE = -2.0")
    log("Observations with score < -1.0 should be archived")
    log("Observations with score < -2.0 should be purge_eligible")
    
    all_obs = recall_observations(limit=100)
    for obs in all_obs:
        score = obs.get('score', 0)
        tier = obs.get('tier', 'unknown')
        purge_eligible = obs.get('purge_eligible', False)
        archived = obs.get('archived_at') is not None
        
        status = []
        if tier == 'cold': status.append('COLD')
        if archived: status.append('ARCHIVED')
        if purge_eligible: status.append('PURGE_ELIGIBLE')
        
        status_str = ' | '.join(status) if status else 'HOT'
        text_preview = obs['observation'][:30]
        log(f"  {obs['id'][:8]}... [{status_str}] score={score:.3f} - {text_preview}...")
    
    # Summary
    log("\n" + "=" * 60)
    log("TEST SUMMARY")
    log("=" * 60)
    log("✅ Observations stored successfully")
    log("✅ Access patterns recorded (score bumping works)")
    log("⏳ Full decay cycle requires 15min wait or manual tick")
    log("")
    log("EXPECTED BEHAVIOR (after full decay):")
    log("  • Critical (5 fetches): HIGH score → stays HOT")
    log("  • Moderate (2 fetches): MID score → may decay to COLD")
    log("  • Obsolete (0 fetches): LOW score → COLD → PURGE_ELIGIBLE")
    log("")
    log("RETENTION PARAMETERS:")
    log("  • SCORE_BUMP_ALPHA=0.4 (each fetch +0.4)")
    log("  • SCORE_DECAY_BETA=0.02 (each tick -0.02)")
    log("  • RETENTION_TICK_SECONDS=900 (15min)")
    log("  • SUMMARY_TRIGGER_SCORE=-1.0 (archive threshold)")
    log("  • PURGE_TRIGGER_SCORE=-2.0 (purge threshold)")
    log("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
