"""
Feature Name Mapper
Maps generic feature names (F1, F2, etc.) to human-readable descriptions
"""

def get_feature_description(feature_name):
    """
    Get human-readable description for a feature
    
    Args:
        feature_name: Generic feature name (e.g., 'F48')
    
    Returns:
        Dictionary with name and description
    """
    
    # Common patterns - categorize features
    feature_num = feature_name.replace('F', '')
    
    try:
        feature_num = int(feature_num)
    except:
        return {
            'display_name': feature_name,
            'category': 'Unknown',
            'description': 'Feature from static analysis'
        }
    
    # Categorize based on feature number ranges
    # (This is inferred - real mapping would come from dataset documentation)
    
    if feature_num <= 100:
        category = 'Permissions'
        description = f'Permission or access control feature #{feature_num}'
        display_name = f'Permission-{feature_num}'
    
    elif 100 < feature_num <= 500:
        category = 'API Calls'
        description = f'API call or method invocation pattern #{feature_num}'
        display_name = f'API-{feature_num}'
    
    elif 500 < feature_num <= 1000:
        category = 'Intents & Components'
        description = f'Intent, activity, service or receiver pattern #{feature_num}'
        display_name = f'Component-{feature_num}'
    
    elif 1000 < feature_num <= 3000:
        category = 'Code Patterns'
        description = f'Code structure or bytecode pattern #{feature_num}'
        display_name = f'Code-{feature_num}'
    
    elif 3000 < feature_num <= 5000:
        category = 'String Patterns'
        description = f'String or constant value pattern #{feature_num}'
        display_name = f'String-{feature_num}'
    
    elif 5000 < feature_num <= 7000:
        category = 'Network & Communication'
        description = f'Network activity or communication pattern #{feature_num}'
        display_name = f'Network-{feature_num}'
    
    elif 7000 < feature_num <= 9000:
        category = 'File & Resources'
        description = f'File structure or resource pattern #{feature_num}'
        display_name = f'Resource-{feature_num}'
    
    else:
        category = 'Behavioral'
        description = f'Behavioral or runtime pattern #{feature_num}'
        display_name = f'Behavior-{feature_num}'
    
    return {
        'display_name': display_name,
        'category': category,
        'description': description,
        'original': feature_name
    }


def enhance_feature_importance_df(feature_importance_df):
    """
    Add descriptions to feature importance dataframe
    
    Args:
        feature_importance_df: DataFrame with 'Feature' column
    
    Returns:
        Enhanced DataFrame with descriptions
    """
    if feature_importance_df is None or feature_importance_df.empty:
        return feature_importance_df
    
    # Add description columns
    descriptions = []
    categories = []
    display_names = []
    
    for feature in feature_importance_df['Feature']:
        info = get_feature_description(feature)
        display_names.append(info['display_name'])
        categories.append(info['category'])
        descriptions.append(info['description'])
    
    # Create enhanced dataframe
    enhanced_df = feature_importance_df.copy()
    enhanced_df.insert(1, 'Display_Name', display_names)
    enhanced_df.insert(2, 'Category', categories)
    enhanced_df.insert(3, 'Description', descriptions)
    
    return enhanced_df


# Specific feature mappings (if we knew the exact mapping)
# This would come from CICANDMAL2020 dataset documentation
KNOWN_FEATURES = {
    'F1': {
        'name': 'Has SEND_SMS Permission',
        'category': 'Permissions',
        'description': 'Application requests permission to send SMS messages',
        'risk': 'High - Common in SMS malware'
    },
    'F8': {
        'name': 'Network Access',
        'category': 'Permissions',
        'description': 'Application requests internet access permission',
        'risk': 'Medium - Required for data exfiltration'
    },
    'F48': {
        'name': 'Sensitive API Usage',
        'category': 'API Calls',
        'description': 'Uses sensitive Android APIs',
        'risk': 'Medium - Potential privacy concern'
    },
    'F58': {
        'name': 'Phone State Access',
        'category': 'Permissions',
        'description': 'Can read phone state and identity',
        'risk': 'Medium - Used for device tracking'
    },
    'F50': {
        'name': 'Location Access',
        'category': 'Permissions',
        'description': 'Requests device location',
        'risk': 'Medium - Privacy concern'
    },
    'F66': {
        'name': 'Storage Access',
        'category': 'Permissions',
        'description': 'Can read/write external storage',
        'risk': 'Medium - Data access'
    },
    'F107': {
        'name': 'Dynamic Code Loading',
        'category': 'Code Patterns',
        'description': 'Uses DexClassLoader or similar',
        'risk': 'High - Common malware technique'
    },
    'F3052': {
        'name': 'Suspicious String Pattern',
        'category': 'String Patterns',
        'description': 'Contains suspicious string constants',
        'risk': 'Medium - Potential malware indicator'
    }
}


def get_known_feature_info(feature_name):
    """
    Get detailed information for known features
    
    Args:
        feature_name: Feature name (e.g., 'F1')
    
    Returns:
        Feature info dict or None
    """
    return KNOWN_FEATURES.get(feature_name, None)




