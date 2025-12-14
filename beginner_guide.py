"""
Beginner-Friendly Explanations for Malware Detection
Makes complex ML concepts easy to understand
"""

# Model Explanations
MODEL_EXPLANATIONS = {
    'Ensemble (Best)': {
        'name': '🎯 Ensemble (Best)',
        'icon': '🎯',
        'simple': 'Combines 3 smart detectors for most accurate results',
        'detailed': '''
        **What it does:**
        - Uses 3 different AI models at once
        - Each model "votes" on whether app is safe or dangerous
        - Combines their opinions for the best answer
        
        **Why it's better:**
        - More accurate than any single model
        - If all 3 agree, you can be very confident
        - Like getting 3 second opinions from doctors
        
        **When to use:**
        ✅ When you need the most accurate result
        ✅ For important decisions
        ✅ Default recommended option
        ''',
        'accuracy': '91-94%',
        'speed': 'Fast (2-3 seconds)',
        'best_for': 'Most accurate overall results'
    },
    
    'LightGBM': {
        'name': '🚀 LightGBM',
        'icon': '🚀',
        'simple': 'Advanced AI that learns from mistakes',
        'detailed': '''
        **What it does:**
        - Uses "gradient boosting" - learns from previous errors
        - Builds decision trees that correct each other's mistakes
        - Very good at catching tricky malware
        
        **Why it's good:**
        - High accuracy (90-92%)
        - Good at finding patterns
        - Industry-standard algorithm
        
        **When to use:**
        ✅ When you want high single-model accuracy
        ✅ For detailed analysis
        ✅ When Ensemble is not needed
        ''',
        'accuracy': '90-92%',
        'speed': 'Fast (1-2 seconds)',
        'best_for': 'Best single model performance'
    },
    
    'Random Forest': {
        'name': '🌲 Random Forest',
        'icon': '🌲',
        'simple': 'Creates hundreds of decision trees that vote together',
        'detailed': '''
        **What it does:**
        - Creates 200 "decision trees" (like flowcharts)
        - Each tree votes: safe or dangerous
        - Majority vote wins
        
        **Why it's reliable:**
        - Very stable and consistent
        - Hard to fool with tricky data
        - Proven track record
        
        **When to use:**
        ✅ When you want a reliable baseline
        ✅ For consistent results
        ✅ When speed matters
        ''',
        'accuracy': '87.64%',
        'speed': 'Very Fast (<1 second)',
        'best_for': 'Reliable, consistent results'
    },
    
    'Logistic Regression': {
        'name': '📊 Logistic Regression',
        'icon': '📊',
        'simple': 'Mathematical model that calculates probability of malware',
        'detailed': '''
        **What it does:**
        - Uses math to calculate risk probability
        - Looks at which features correlate with malware
        - Fast and efficient
        
        **Why it's useful:**
        - Very fast predictions
        - Easy to understand how it works
        - Good for simple cases
        
        **When to use:**
        ✅ When you need fastest results
        ✅ For quick screening
        ✅ When processing many apps
        ''',
        'accuracy': '86.63%',
        'speed': 'Ultra Fast (<0.5 seconds)',
        'best_for': 'Speed over accuracy'
    }
}

# Feature Category Explanations
FEATURE_CATEGORIES = {
    'Permissions': {
        'icon': '🔐',
        'simple': 'What the app is allowed to access on your phone',
        'detailed': '''
        **What are permissions?**
        Permissions are like keys that let apps access your phone's features.
        
        **Examples:**
        - 📷 CAMERA: Can take photos/videos
        - 📍 LOCATION: Can see where you are
        - 📱 READ_SMS: Can read your text messages
        - 📞 CALL_PHONE: Can make phone calls
        
        **Why it matters:**
        Malware often asks for suspicious permission combinations.
        Example: Why would a flashlight app need to read your SMS?
        
        **Red flags:**
        ⚠️ Too many permissions (normal apps use 5-10)
        ⚠️ Dangerous combinations (SMS + Internet + Location)
        ⚠️ Permissions that don't match app purpose
        ''',
        'examples': [
            '✅ Camera app requesting CAMERA permission = Normal',
            '⚠️ Calculator app requesting READ_SMS = Suspicious!',
            '🚨 Flashlight requesting LOCATION + INTERNET = Very suspicious!'
        ]
    },
    
    'API Calls': {
        'icon': '⚙️',
        'simple': 'Internal code functions the app uses',
        'detailed': '''
        **What are API calls?**
        These are programming commands the app uses internally.
        Think of them as "actions" the app can perform.
        
        **Examples:**
        - HttpURLConnection: Connect to internet
        - Runtime.exec: Run system commands
        - DexClassLoader: Load additional code
        - TelephonyManager: Access phone info
        
        **Why it matters:**
        Malware uses suspicious API calls to hide or do harmful things.
        
        **Red flags:**
        ⚠️ Runtime.exec: Can run hidden commands
        ⚠️ DexClassLoader: Can download more malicious code
        ⚠️ Cipher.getInstance: Might encrypt your files (ransomware)
        ⚠️ sendTextMessage: Can send SMS to premium numbers
        ''',
        'examples': [
            '✅ HttpURLConnection in a web browser = Normal',
            '⚠️ DexClassLoader in any app = Suspicious!',
            '🚨 Runtime.exec + sendTextMessage = Very dangerous!'
        ]
    },
    
    'App Components': {
        'icon': '🧩',
        'simple': 'Building blocks that make up the app',
        'detailed': '''
        **What are components?**
        Android apps are made of 4 main parts:
        
        1. **Activities** 📱
           - Screens you see and interact with
           - Example: Login screen, main menu
           - Normal apps have 5-20 activities
        
        2. **Services** ⚙️
           - Background tasks that run invisibly
           - Example: Playing music while app is minimized
           - Malware uses services to hide
        
        3. **Receivers** 📡
           - Listen for system events
           - Example: Detect when phone boots up
           - Malware uses to restart after reboot
        
        4. **Providers** 💾
           - Share data between apps
           - Example: Contact information
           - Can be exploited by malware
        
        **Red flags:**
        ⚠️ Too many services (malware hiding in background)
        ⚠️ Boot receivers (malware that restarts)
        ⚠️ More services than activities (suspicious ratio)
        ''',
        'examples': [
            '✅ 10 activities, 2 services = Normal app',
            '⚠️ 2 activities, 10 services = Hiding something!',
            '🚨 BOOT_COMPLETED receiver + many services = Malware!'
        ]
    },
    
    'File Structure': {
        'icon': '📁',
        'simple': 'How the app\'s files are organized internally',
        'detailed': '''
        **What is file structure?**
        Apps are like ZIP files containing code and resources.
        
        **Key files:**
        - **DEX files**: Actual app code (normally 1-2 files)
        - **Resources**: Images, sounds, layouts
        - **Libraries**: Reusable code pieces
        - **Manifest**: App configuration
        
        **Why it matters:**
        Malware often has unusual file structures.
        
        **Red flags:**
        ⚠️ Multiple DEX files: Might be hiding code
        ⚠️ Too many files: Might be repacked malware
        ⚠️ Suspicious file names: Hidden payloads
        ⚠️ Encrypted resources: Hiding malicious content
        ''',
        'examples': [
            '✅ 1 DEX file, 200 resources = Normal',
            '⚠️ 5 DEX files = Code splitting (suspicious)',
            '🚨 Encrypted DEX files = Definitely hiding something!'
        ]
    },
    
    'Network Activity': {
        'icon': '🌐',
        'simple': 'Where the app sends/receives data on the internet',
        'detailed': '''
        **What is network activity?**
        Apps often communicate with servers on the internet.
        
        **Normal uses:**
        - Check for updates
        - Load content
        - Sync data
        - Ads
        
        **Malware uses:**
        - Send your personal data to hackers
        - Download more malicious code
        - Connect to command & control servers
        - Send SMS to premium numbers
        
        **Red flags:**
        ⚠️ Suspicious domains (.tk, .ml, .ga = free domains)
        ⚠️ IP addresses instead of names (trying to hide)
        ⚠️ Many different servers (command & control)
        ⚠️ Encrypted traffic without good reason
        ''',
        'examples': [
            '✅ Connecting to google.com = Normal',
            '⚠️ Connecting to 123.45.67.89 = Suspicious!',
            '🚨 Connecting to freehacks.tk = Malware!'
        ]
    },
    
    'Behavior Patterns': {
        'icon': '🎭',
        'simple': 'Suspicious combinations of features that indicate malware',
        'detailed': '''
        **What are behavior patterns?**
        It's not just individual features - it's how they combine.
        Like a doctor looking at multiple symptoms together.
        
        **Common malware patterns:**
        
        1. **Spyware Pattern** 🕵️
           - LOCATION + CAMERA + INTERNET
           - Records where you are and what you do
           - Sends data to attacker
        
        2. **SMS Malware Pattern** 💸
           - READ_SMS + SEND_SMS + INTERNET
           - Steals banking codes from SMS
           - Sends expensive texts
        
        3. **Ransomware Pattern** 🔒
           - WRITE_STORAGE + ADMIN_PERMISSION
           - Encrypts your files
           - Demands payment
        
        4. **Banking Trojan** 💰
           - SMS + Internet + Screen capture
           - Steals banking credentials
           - Transfers money
        
        **Red flags:**
        ⚠️ Any of these patterns = High malware risk!
        ''',
        'examples': [
            '✅ Camera app: CAMERA only = Normal',
            '⚠️ Flashlight: LOCATION + INTERNET = Why?!',
            '🚨 Game: READ_SMS + SEND_SMS + CONTACTS = Malware!'
        ]
    }
}

# Confidence Level Explanations
CONFIDENCE_EXPLANATIONS = {
    'high': {
        'level': 'High Confidence (80%+)',
        'meaning': 'The AI is very sure about this result',
        'details': '''
        **What this means:**
        - All detection models strongly agree
        - Clear patterns detected
        - Similar to apps we've seen before
        
        **You can trust this result:**
        ✅ 95% of high-confidence predictions are correct
        ✅ Safe to make decisions based on this
        ✅ No second opinion needed
        
        **What to do:**
        - If "Malware": DO NOT INSTALL! 🚫
        - If "Benign": Safe to use ✅
        ''',
        'icon': '🎯',
        'color': 'green'
    },
    
    'medium': {
        'level': 'Medium Confidence (60-80%)',
        'meaning': 'The AI thinks this is likely correct, but not 100% sure',
        'details': '''
        **What this means:**
        - Models somewhat agree
        - Some patterns detected, but not all clear
        - Might need more analysis
        
        **You should be cautious:**
        ⚠️ 70-80% of medium-confidence predictions are correct
        ⚠️ Consider getting more information
        ⚠️ Look at individual features
        
        **What to do:**
        - If "Malware": Avoid unless you trust the source
        - If "Benign": Probably safe, but check permissions
        - Read the detailed analysis below
        ''',
        'icon': '⚡',
        'color': 'orange'
    },
    
    'low': {
        'level': 'Low Confidence (<60%)',
        'meaning': 'The AI is unsure - this is a difficult case',
        'details': '''
        **What this means:**
        - Models disagree with each other
        - Unusual patterns (might be new malware or rare app)
        - Not enough clear evidence
        
        **Be very careful:**
        ⚠️ Only 50-60% of low-confidence predictions are correct
        ⚠️ This is basically a guess
        ⚠️ Don't make important decisions on this alone
        
        **What to do:**
        - Upload to VirusTotal for more scans
        - Check developer reputation
        - Read all detailed analysis
        - When in doubt, don't install!
        ''',
        'icon': '❓',
        'color': 'red'
    }
}

# Malware Type Explanations
MALWARE_TYPES = {
    'Adware': {
        'icon': '📢',
        'simple': 'Shows unwanted ads and pop-ups',
        'danger': 'Low to Medium',
        'description': '''
        **What it does:**
        - Displays lots of annoying advertisements
        - Sometimes hard to close or remove
        - May track your browsing habits
        - Slows down your phone
        
        **How harmful:**
        🟡 Usually not very dangerous
        - Won't steal your data (usually)
        - Won't damage your device
        - Just very annoying
        
        **What to do:**
        ⚠️ Avoid installing
        ✅ If installed, uninstall immediately
        ✅ Clear cache and data
        ''',
        'examples': 'Apps that show full-screen ads, fake "virus detected" alerts'
    },
    
    'Trojan': {
        'icon': '🐴',
        'simple': 'Pretends to be useful but does harmful things secretly',
        'danger': 'High',
        'description': '''
        **What it does:**
        - Looks like a normal app (game, utility, etc.)
        - Secretly steals your data
        - Can download more malware
        - Opens backdoors for hackers
        
        **How harmful:**
        🔴 Very dangerous!
        - Can steal passwords, photos, messages
        - May access banking apps
        - Hard to detect
        
        **What to do:**
        🚫 DO NOT INSTALL!
        ⚠️ If installed, factory reset recommended
        ⚠️ Change all passwords
        ⚠️ Check bank statements
        ''',
        'examples': 'Fake YouTube, fake WhatsApp, "system update" apps'
    },
    
    'Spyware': {
        'icon': '🕵️',
        'simple': 'Secretly watches and records everything you do',
        'danger': 'Very High',
        'description': '''
        **What it does:**
        - Records your location
        - Captures screenshots
        - Records audio/video
        - Steals messages and calls
        - Sends everything to attacker
        
        **How harmful:**
        🔴 Extremely dangerous!
        - Total privacy invasion
        - Can blackmail you
        - Stalkerware (used by abusers)
        
        **What to do:**
        🚫 NEVER INSTALL!
        ⚠️ If found: Factory reset immediately
        ⚠️ Report to authorities if stalked
        ⚠️ Seek help if in danger
        ''',
        'examples': 'Stalkerware, keyloggers, screen recorders'
    },
    
    'Ransomware': {
        'icon': '🔒',
        'simple': 'Locks your files and demands money to unlock them',
        'danger': 'Critical',
        'description': '''
        **What it does:**
        - Encrypts all your files
        - Makes photos/documents inaccessible
        - Demands payment (usually Bitcoin)
        - May not unlock even if you pay!
        
        **How harmful:**
        🔴 EXTREMELY DANGEROUS!
        - Can lose all your data forever
        - May spread to connected devices
        - Financial loss
        
        **What to do:**
        🚫 NEVER EVER INSTALL!
        ⚠️ Keep backups of important files
        ⚠️ Don't pay the ransom (doesn't guarantee return)
        ⚠️ Seek professional help if infected
        ''',
        'examples': 'CryptoLocker, WannaCry variants for Android'
    },
    
    'Banking Trojan': {
        'icon': '💰',
        'simple': 'Steals your money by accessing banking apps',
        'danger': 'Critical',
        'description': '''
        **What it does:**
        - Overlays fake login screens on banking apps
        - Steals your banking credentials
        - Intercepts SMS codes (2FA)
        - Transfers money from your account
        
        **How harmful:**
        🔴 CRITICAL DANGER!
        - Direct financial theft
        - Can empty bank accounts
        - Steals credit card info
        
        **What to do:**
        🚫 ABSOLUTELY DO NOT INSTALL!
        ⚠️ If infected: Contact bank immediately
        ⚠️ Freeze accounts
        ⚠️ Factory reset device
        ⚠️ File police report
        ''',
        'examples': 'Anubis, Cerberus, BankBot'
    },
    
    'SMS Malware': {
        'icon': '📱',
        'simple': 'Sends expensive text messages without you knowing',
        'danger': 'High',
        'description': '''
        **What it does:**
        - Sends SMS to premium numbers
        - Each text costs $5-20
        - Can rack up hundreds in charges
        - Steals SMS verification codes
        
        **How harmful:**
        🔴 Very dangerous financially!
        - Expensive phone bills
        - Steals 2-factor authentication codes
        - Can access accounts
        
        **What to do:**
        🚫 Do not install!
        ⚠️ Monitor phone bill for unusual charges
        ⚠️ Contact carrier if suspicious charges
        ⚠️ Uninstall and scan device
        ''',
        'examples': 'FakeInst, Boxer SMS'
    },
    
    'Benign': {
        'icon': '✅',
        'simple': 'Safe and harmless application',
        'danger': 'None',
        'description': '''
        **What it means:**
        - Normal, legitimate application
        - No malicious behavior detected
        - Safe to use
        
        **This is good!**
        ✅ No threats detected
        ✅ Appropriate permissions
        ✅ Normal behavior patterns
        ✅ Safe to install
        
        **Still be smart:**
        - Only install from trusted sources (Google Play)
        - Check user reviews
        - Review permissions before installing
        - Keep app updated
        ''',
        'examples': 'Official apps from trusted developers'
    }
}


def get_beginner_explanation(category, item=None):
    """Get beginner-friendly explanation for a category or item"""
    if category == 'model' and item:
        return MODEL_EXPLANATIONS.get(item, {})
    elif category == 'feature_category':
        return FEATURE_CATEGORIES
    elif category == 'confidence' and item:
        return CONFIDENCE_EXPLANATIONS.get(item, {})
    elif category == 'malware_type' and item:
        return MALWARE_TYPES.get(item, MALWARE_TYPES.get('Benign', {}))
    return {}


def get_simple_explanation(technical_term):
    """Convert technical term to simple language"""
    simple_terms = {
        'API': 'Programming command',
        'DEX': 'Android code file',
        'Manifest': 'App configuration',
        'Intent': 'Message between app parts',
        'Service': 'Background task',
        'Activity': 'Screen',
        'Receiver': 'Event listener',
        'Provider': 'Data sharer',
        'Permission': 'Access right',
        'Entropy': 'Randomness (high = encrypted/hidden)',
        'Obfuscation': 'Code hiding technique',
        'Certificate': 'Developer signature',
        'SHA256': 'Unique fingerprint',
        'Ensemble': 'Multiple models voting together',
        'Gradient Boosting': 'Learning from mistakes',
        'Random Forest': 'Many decision trees voting',
        'Logistic Regression': 'Mathematical probability',
        'Confidence': 'How sure the AI is',
        'Accuracy': 'How often it gets it right',
        'F1-Score': 'Balance of precision and recall',
        'Precision': 'When it says malware, how often is it right',
        'Recall': 'How many malware does it catch',
    }
    
    return simple_terms.get(technical_term, technical_term)

