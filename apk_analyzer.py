"""
Comprehensive APK Feature Extractor for Malware Detection
Extracts static analysis features from APK/XAPK files
"""

import numpy as np
import pandas as pd
from pathlib import Path
import zipfile
import tempfile
import os
import hashlib
from collections import Counter

try:
    from androguard.core import apk as androguard_apk
    from androguard.core.dex import DEX
    from androguard.core.analysis.analysis import Analysis
    ANDROGUARD_AVAILABLE = True
except ImportError:
    ANDROGUARD_AVAILABLE = False
    print("⚠️ Androguard not available. Install with: pip install androguard")
    print("⚠️ Androguard not available. Install with: pip install androguard")


class APKAnalyzer:
    """
    Comprehensive APK analyzer for malware detection
    Extracts features matching static analysis patterns
    """
    
    def __init__(self, feature_columns):
        """
        Initialize analyzer
        
        Args:
            feature_columns: List of expected feature names (9,503 features)
        """
        self.feature_columns = feature_columns
        self.n_features = len(feature_columns)
        
        # Android permissions (commonly abused by malware)
        self.dangerous_permissions = self._get_dangerous_permissions()
        self.api_calls = self._get_suspicious_api_calls()
        
    def extract_from_file(self, file_path, file_type='apk'):
        """
        Extract features from APK or XAPK file
        
        Args:
            file_path: Path to file
            file_type: 'apk' or 'xapk'
            
        Returns:
            Dictionary of extracted features
        """
        if file_type == 'xapk':
            # Extract APK from XAPK
            apk_path = self._extract_apk_from_xapk(file_path)
            if apk_path is None:
                return None
        else:
            apk_path = file_path
        
        # Extract features
        return self._analyze_apk(apk_path)
    
    def _extract_apk_from_xapk(self, xapk_path):
        """Extract base APK from XAPK archive"""
        try:
            temp_dir = tempfile.mkdtemp()
            
            with zipfile.ZipFile(xapk_path, 'r') as xapk:
                # Find APK files
                apk_files = [f for f in xapk.namelist() if f.endswith('.apk')]
                
                if not apk_files:
                    print("No APK found in XAPK")
                    return None
                
                # Extract first APK
                xapk.extract(apk_files[0], temp_dir)
                return os.path.join(temp_dir, apk_files[0])
                
        except Exception as e:
            print(f"Error extracting XAPK: {e}")
            return None
    
    def _analyze_apk(self, apk_path):
        """
        Main APK analysis function
        Extracts comprehensive features
        """
        features = {}
        
        if not ANDROGUARD_AVAILABLE:
            return self._extract_basic_features(apk_path)
        
        try:
            # Load APK with Androguard
            apk = androguard_apk.APK(apk_path)
            
            print(f"📱 Analyzing: {apk.get_package()}")
            print(f"   Version: {apk.get_androidversion_name()}")
            
            # Extract different feature categories
            print("   Extracting permissions...")
            perm_features = self._extract_permissions(apk)
            features.update(perm_features)
            print(f"   ✓ Permissions: {perm_features.get('n_permissions', 0)}")
            
            print("   Extracting intents...")
            intent_features = self._extract_intents(apk)
            features.update(intent_features)
            print(f"   ✓ Activities: {intent_features.get('n_activities', 0)}")
            
            print("   Extracting API calls...")
            api_features = self._extract_api_calls(apk)
            features.update(api_features)
            print(f"   ✓ API calls: {len(api_features.get('api_calls', []))}")
            
            print("   Extracting file info...")
            features.update(self._extract_file_info(apk, apk_path))
            
            print("   Extracting certificate...")
            features.update(self._extract_certificate_info(apk))
            
            print("   Extracting strings...")
            features.update(self._extract_strings(apk))
            
            print(f"✅ Total features extracted: {len(features)}")
            
            return features
            
        except Exception as e:
            print(f"❌ Error analyzing APK: {e}")
            import traceback
            traceback.print_exc()
            return self._extract_basic_features(apk_path)
    
    def _extract_permissions(self, apk):
        """Extract permission-based features"""
        features = {}
        
        permissions = apk.get_permissions()
        
        # Store actual permission list for display
        features['permissions'] = permissions
        
        # Check each dangerous permission
        for perm in self.dangerous_permissions:
            perm_key = f"permission_{perm.split('.')[-1]}"
            features[perm_key] = 1 if perm in permissions else 0
        
        # Count permission categories
        features['n_permissions'] = len(permissions)
        features['n_dangerous_permissions'] = sum(
            1 for p in permissions if p in self.dangerous_permissions
        )
        
        return features
    
    def _extract_intents(self, apk):
        """Extract intent-based features"""
        features = {}
        
        # Store actual lists for display
        features['activities'] = apk.get_activities()
        features['services'] = apk.get_services()
        features['receivers'] = apk.get_receivers()
        features['providers'] = apk.get_providers()
        
        # Activities, services, receivers, providers (counts for ML)
        features['n_activities'] = len(apk.get_activities())
        features['n_services'] = len(apk.get_services())
        features['n_receivers'] = len(apk.get_receivers())
        features['n_providers'] = len(apk.get_providers())
        
        # Check for exported components (security risk)
        features['has_exported_activity'] = 0
        features['has_exported_service'] = 0
        features['has_exported_receiver'] = 0
        
        try:
            manifest = apk.get_android_manifest_axml()
            if manifest:
                # Parse manifest for exported flags
                pass  # Simplified for now
        except:
            pass
        
        return features
    
    def _extract_api_calls(self, apk):
        """Extract API call patterns"""
        features = {}
        
        try:
            # Get DEX files
            dex_files = apk.get_all_dex()
            
            api_call_count = Counter()
            
            for dex_data in dex_files:
                try:
                    dex = DEX(dex_data)
                    
                    # Analyze methods
                    for method in dex.get_methods():
                        method_str = str(method.get_name())
                        # Count suspicious API calls
                        for api in self.api_calls:
                            if api in method_str:
                                api_call_count[api] += 1
                except Exception as e:
                    print(f"Warning: Could not analyze DEX file: {e}")
                    continue
            
            # Store actual API call list for display
            features['api_calls'] = [api for api, count in api_call_count.most_common(50)]
            
            # Add API call features (for ML)
            for api in self.api_calls[:50]:  # Top 50 APIs
                api_key = f"api_{api.replace('.', '_')}"
                features[api_key] = api_call_count.get(api, 0)
            
            features['total_api_calls'] = sum(api_call_count.values())
            
        except Exception as e:
            print(f"Warning: Could not extract API calls: {e}")
            features['api_calls'] = []
            features['total_api_calls'] = 0
        
        return features
    
    def _extract_file_info(self, apk, apk_path):
        """Extract file structure features"""
        features = {}
        
        # APK size
        features['apk_size'] = os.path.getsize(apk_path)
        
        # File counts (for both display and ML)
        features['file_count'] = len(apk.get_files())  # For display
        features['n_files'] = len(apk.get_files())     # For ML
        dex_names = list(apk.get_dex_names())  # Convert filter to list
        features['dex_files'] = len(dex_names)  # For display
        features['n_dex_files'] = len(dex_names)  # For ML
        
        # Count file types
        files = apk.get_files()
        features['n_so_files'] = len([f for f in files if f.endswith('.so')])
        features['n_png_files'] = len([f for f in files if f.endswith('.png')])
        features['n_xml_files'] = len([f for f in files if f.endswith('.xml')])
        
        # Check for suspicious files
        features['has_classes_dex'] = 1 if 'classes.dex' in files else 0
        features['has_native_code'] = 1 if any(f.endswith('.so') for f in files) else 0
        
        return features
    
    def _extract_certificate_info(self, apk):
        """Extract certificate/signature features"""
        features = {}
        
        try:
            cert = apk.get_certificate(apk.get_certificates()[0])
            
            features['cert_serial'] = int(cert.serial_number) % 10000  # Normalized
            features['cert_version'] = cert.version
            
            # Check certificate validity
            import datetime
            now = datetime.datetime.now()
            features['cert_is_valid'] = 1 if (
                cert.not_valid_before <= now <= cert.not_valid_after
            ) else 0
            
        except:
            features['cert_serial'] = 0
            features['cert_version'] = 0
            features['cert_is_valid'] = 0
        
        return features
    
    def _extract_strings(self, apk):
        """Extract string-based features"""
        features = {}
        
        try:
            # Get all strings from DEX
            strings = []
            for dex_data in apk.get_all_dex():
                dex = dvm.DalvikVMFormat(dex_data)
                strings.extend(dex.get_strings())
            
            features['n_strings'] = len(strings)
            
            # Check for suspicious strings
            suspicious_keywords = [
                'root', 'su', 'superuser', 'password', 'credit', 'card',
                'bank', 'sms', 'premium', 'hack', 'crack', 'patch'
            ]
            
            strings_lower = [s.lower() for s in strings]
            for keyword in suspicious_keywords:
                key = f"string_{keyword}"
                features[key] = sum(1 for s in strings_lower if keyword in s)
            
        except:
            features['n_strings'] = 0
        
        return features
    
    def _extract_basic_features(self, apk_path):
        """Fallback: basic features without Androguard"""
        features = {}
        
        try:
            with zipfile.ZipFile(apk_path, 'r') as apk:
                files = apk.namelist()
                
                features['n_files'] = len(files)
                features['n_so_files'] = len([f for f in files if f.endswith('.so')])
                features['n_png_files'] = len([f for f in files if f.endswith('.png')])
                features['n_xml_files'] = len([f for f in files if f.endswith('.xml')])
                features['has_classes_dex'] = 1 if 'classes.dex' in files else 0
                features['apk_size'] = os.path.getsize(apk_path)
                
        except:
            pass
        
        return features
    
    def features_to_vector(self, features_dict):
        """
        Convert extracted features to feature vector
        Maps to the 9,503 feature format
        
        Args:
            features_dict: Dictionary of extracted features
            
        Returns:
            numpy array (1, 9503)
        """
        # Initialize feature vector
        feature_vector = np.zeros(self.n_features)
        
        # Map extracted features to feature vector
        # This is a simplified mapping - in production, you'd need
        # exact mapping from CICANDMAL2020 dataset documentation
        
        feature_idx = 0
        for key, value in sorted(features_dict.items()):
            if feature_idx < self.n_features:
                # Normalize large values
                if isinstance(value, (int, float)):
                    if value > 1000:
                        value = np.log1p(value)  # Log scale for large values
                    feature_vector[feature_idx] = min(value, 1.0)
                else:
                    feature_vector[feature_idx] = 1.0 if value else 0.0
                feature_idx += 1
        
        return feature_vector.reshape(1, -1)
    
    def _get_dangerous_permissions(self):
        """List of dangerous Android permissions"""
        return [
            'android.permission.SEND_SMS',
            'android.permission.RECEIVE_SMS',
            'android.permission.READ_SMS',
            'android.permission.WRITE_SMS',
            'android.permission.RECEIVE_MMS',
            'android.permission.READ_CONTACTS',
            'android.permission.WRITE_CONTACTS',
            'android.permission.READ_PHONE_STATE',
            'android.permission.CALL_PHONE',
            'android.permission.READ_CALL_LOG',
            'android.permission.WRITE_CALL_LOG',
            'android.permission.ACCESS_FINE_LOCATION',
            'android.permission.ACCESS_COARSE_LOCATION',
            'android.permission.CAMERA',
            'android.permission.RECORD_AUDIO',
            'android.permission.READ_EXTERNAL_STORAGE',
            'android.permission.WRITE_EXTERNAL_STORAGE',
            'android.permission.RECEIVE_BOOT_COMPLETED',
            'android.permission.SYSTEM_ALERT_WINDOW',
            'android.permission.GET_ACCOUNTS',
            'android.permission.WAKE_LOCK',
            'android.permission.INTERNET',
            'android.permission.ACCESS_NETWORK_STATE',
            'android.permission.CHANGE_WIFI_STATE',
            'android.permission.BLUETOOTH',
            'android.permission.BLUETOOTH_ADMIN',
            'android.permission.READ_CALENDAR',
            'android.permission.WRITE_CALENDAR',
            'android.permission.INSTALL_PACKAGES',
            'android.permission.DELETE_PACKAGES',
            'android.permission.PROCESS_OUTGOING_CALLS',
        ]
    
    def _get_suspicious_api_calls(self):
        """List of suspicious API calls"""
        return [
            'sendTextMessage',
            'getDeviceId',
            'getSubscriberId',
            'getSimSerialNumber',
            'getLine1Number',
            'Runtime.exec',
            'ProcessBuilder',
            'System.loadLibrary',
            'DexClassLoader',
            'PathClassLoader',
            'HttpURLConnection',
            'TelephonyManager',
            'SmsManager',
            'abortBroadcast',
            'getLastKnownLocation',
            'getAccounts',
            'setWifiEnabled',
            'sendBroadcast',
            'android.intent.action.BOOT_COMPLETED',
            'android.intent.action.PACKAGE_ADDED',
        ]


# Test function
if __name__ == "__main__":
    import joblib
    
    # Load feature columns
    feature_columns = joblib.load("models/feature_columns.pkl")
    
    # Create analyzer
    analyzer = APKAnalyzer(feature_columns)
    
    print(f"✅ APK Analyzer initialized")
    print(f"   Expected features: {analyzer.n_features}")
    print(f"   Androguard available: {ANDROGUARD_AVAILABLE}")


