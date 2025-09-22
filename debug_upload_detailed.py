#!/usr/bin/env python3
"""
Detailed debugging script to check upload form submission issue
"""
import requests
import os

def test_upload_debug():
    # Test image path
    test_image = "uploads/20250922_142357_cow.jpg"
    
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return
    
    print(f"🧪 Testing upload with: {test_image}")
    
    # Create session to maintain cookies
    session = requests.Session()
    
    # First get the homepage to establish session
    print("1️⃣ Getting homepage...")
    home_response = session.get("http://127.0.0.1:5000/")
    print(f"   Homepage status: {home_response.status_code}")
    
    # Now test the upload
    print("2️⃣ Testing upload...")
    with open(test_image, 'rb') as img_file:
        files = {'file': (os.path.basename(test_image), img_file, 'image/jpeg')}
        
        # Test upload with detailed debugging
        upload_response = session.post(
            "http://127.0.0.1:5000/upload",
            files=files,
            allow_redirects=False  # Don't follow redirects automatically
        )
        
        print(f"   Upload status: {upload_response.status_code}")
        print(f"   Upload headers: {dict(upload_response.headers)}")
        
        if upload_response.status_code == 302:
            redirect_location = upload_response.headers.get('Location', 'Unknown')
            print(f"   Redirect to: {redirect_location}")
            
            # Follow the redirect manually
            if redirect_location:
                print("3️⃣ Following redirect...")
                if redirect_location.startswith('/'):
                    redirect_url = f"http://127.0.0.1:5000{redirect_location}"
                else:
                    redirect_url = redirect_location
                
                final_response = session.get(redirect_url)
                print(f"   Final status: {final_response.status_code}")
                print(f"   Final URL: {final_response.url}")
                
                # Check if we ended up on results page
                if 'results' in final_response.url:
                    print("✅ Successfully redirected to results page!")
                elif final_response.url.endswith('/'):
                    print("❌ Redirected back to homepage - there's an issue!")
                    
                    # Let's check the content to see what happened
                    if "Drag & Drop Your Cattle Image" in final_response.text:
                        print("   Confirmed: We're back on the upload page")
                else:
                    print(f"   Unexpected redirect to: {final_response.url}")
        
        elif upload_response.status_code == 200:
            print("✅ Upload successful (200)")
        else:
            print(f"❌ Upload failed with status: {upload_response.status_code}")
            print(f"   Response: {upload_response.text[:200]}...")

if __name__ == "__main__":
    test_upload_debug()