import pexpect
import sys

child = pexpect.spawn('gh auth login -h github.com -p https -w', encoding='utf-8')

# Wait for "First copy your one-time code: "
try:
    child.expect(r'First copy your one-time code: ([A-Z0-9-]+)', timeout=10)
    code = child.match.group(1)
    print(f"CODE: {code}")
    print("URL: https://github.com/login/device")
    
    # Send Enter to open the browser
    child.sendline("")
    
    # Wait for authentication to complete
    child.expect('Logged in as', timeout=120)
    print(child.after)
except pexpect.TIMEOUT:
    print("Timeout waiting for gh auth output")
    print("Output so far:", child.before)
