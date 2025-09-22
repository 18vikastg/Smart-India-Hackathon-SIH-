#!/usr/bin/env python3
"""
Test form submission exactly like browser does
"""
import requests
import os

def test_browser_like_upload():
    test_image = "uploads/20250922_142357_cow.jpg"
    
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return
    
    print(f"🧪 Testing browser-like upload with: {test_image}")
    
    # Create session like browser
    session = requests.Session()
    
    # Get homepage first (like browser)
    print("1️⃣ Loading homepage...")
    home_response = session.get("http://127.0.0.1:5000/")
    print(f"   Status: {home_response.status_code}")
    
    # Extract any CSRF tokens or session info if needed
    # (Flask apps sometimes have this)
    
    # Now submit form exactly like browser would
    print("2️⃣ Submitting form...")
    
    with open(test_image, 'rb') as img_file:
        # Prepare form data exactly like HTML form
        files = {
            'file': (os.path.basename(test_image), img_file, 'image/jpeg')
        }
        
        # Headers that browser typically sends
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # Submit to /upload with POST (exactly like the form)
        response = session.post(
            "http://127.0.0.1:5000/upload",
            files=files,
            headers=headers,
            allow_redirects=True  # Follow redirects like browser
        )
        
        print(f"   Final status: {response.status_code}")
        print(f"   Final URL: {response.url}")
        print(f"   Response length: {len(response.text)}")
        
        # Check what page we ended up on
        if 'results' in response.url:
            print("✅ SUCCESS: Ended up on results page!")
            if 'breed' in response.text.lower():
                print("✅ Results page contains breed information!")
            else:
                print("⚠️  Results page loaded but no breed info found")
        elif response.url.endswith('/') or 'Drag & Drop' in response.text:
            print("❌ ISSUE: Redirected back to upload page")
        else:
            print(f"❓ Unexpected result: {response.url}")
            
        # Print a snippet of the response
        print(f"\n📄 Response snippet:")
        print(response.text[:300] + "..." if len(response.text) > 300 else response.text)

if __name__ == "__main__":
    test_browser_like_upload()