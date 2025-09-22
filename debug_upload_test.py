#!/usr/bin/env python3
"""
Debug Upload Testing Script
"""

import requests
import os
import time

def test_upload_with_debug():
    print("🧪 Testing Upload Functionality with Debug Info")
    print("=" * 50)
    
    # Test image path
    test_image = "uploads/20250922_142357_cow.jpg"
    
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return
    
    print(f"📸 Using test image: {test_image}")
    print(f"📏 File size: {os.path.getsize(test_image)} bytes")
    
    # Test 1: Check homepage accessibility
    print("\n🏠 Testing Homepage...")
    try:
        response = requests.get("http://127.0.0.1:5000", timeout=10)
        print(f"✅ Homepage Status: {response.status_code}")
        if "Indian Cattle" in response.text:
            print("✅ Homepage loaded correctly with content")
        else:
            print("⚠️ Homepage loaded but content might be missing")
    except Exception as e:
        print(f"❌ Homepage error: {e}")
        return
    
    # Test 2: Test file upload via form
    print("\n📤 Testing File Upload...")
    try:
        with open(test_image, 'rb') as f:
            files = {'file': (os.path.basename(test_image), f, 'image/jpeg')}
            
            # Simulate form POST to /upload
            response = requests.post(
                "http://127.0.0.1:5000/upload", 
                files=files,
                timeout=30,
                allow_redirects=False  # Don't follow redirects to see what happens
            )
            
        print(f"🔄 Upload Status: {response.status_code}")
        print(f"📍 Response Headers: {dict(response.headers)}")
        
        if response.status_code == 302:
            redirect_url = response.headers.get('Location', 'No location header')
            print(f"↗️ Redirect to: {redirect_url}")
            
            # Follow the redirect manually
            if redirect_url.startswith('/'):
                redirect_url = "http://127.0.0.1:5000" + redirect_url
            
            print(f"\n🔍 Following redirect to: {redirect_url}")
            redirect_response = requests.get(redirect_url, timeout=10)
            print(f"✅ Results Page Status: {redirect_response.status_code}")
            
            if redirect_response.status_code == 200:
                if "breed" in redirect_response.text.lower():
                    print("✅ Results page loaded with breed information!")
                    # Check for specific result elements
                    if "confidence" in redirect_response.text.lower():
                        print("✅ Confidence score found in results")
                    if "analysis" in redirect_response.text.lower():
                        print("✅ Analysis information found in results")
                else:
                    print("⚠️ Results page loaded but breed info might be missing")
            else:
                print(f"❌ Results page failed with status: {redirect_response.status_code}")
                
        elif response.status_code == 200:
            print("✅ Upload successful - got direct response")
            if "breed" in response.text.lower():
                print("✅ Breed information found in response")
        else:
            print(f"❌ Upload failed with status: {response.status_code}")
            print(f"📝 Response content (first 200 chars): {response.text[:200]}")
            
    except Exception as e:
        print(f"❌ Upload test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("🎯 Debug Test Complete!")

if __name__ == "__main__":
    test_upload_with_debug()