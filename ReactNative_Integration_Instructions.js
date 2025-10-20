// Integration Guide for Adding Family Recognition to Existing React Native App
// Using IP: 10.200.253.165

/**
 * STEP 1: Install Required Dependencies
 */
// Run in your React Native project terminal:
// npm install expo-camera expo-media-library expo-file-system @react-native-picker/picker
// or
// yarn add expo-camera expo-media-library expo-file-system @react-native-picker/picker

/**
 * STEP 2: Add Navigation Routes
 * If using React Navigation, add these screens to your navigation:
 */

// In your main navigator (e.g., App.js or navigator file):
import AddFamilyMemberScreen from './screens/AddFamilyMemberScreen';
import WhoIsThisScreen from './screens/WhoIsThisScreen';
import FamilyListScreen from './screens/FamilyListScreen';

// Add to your stack navigator:
const Stack = createStackNavigator();

function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator>
        {/* Your existing screens */}
        <Stack.Screen name="Home" component={HomeScreen} />
        
        {/* Add these new screens */}
        <Stack.Screen 
          name="WhoIsThis" 
          component={WhoIsThisScreen}
          options={{ title: 'Who Am I?' }}
        />
        <Stack.Screen 
          name="AddFamilyMember" 
          component={AddFamilyMemberScreen}
          options={{ title: 'Add Family Member' }}
        />
        <Stack.Screen 
          name="FamilyList" 
          component={FamilyListScreen}
          options={{ title: 'Family Members' }}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
}

/**
 * STEP 3: Add "Who Am I?" Button to Your Home Screen
 */

// In your existing HomeScreen.js, add this button:
import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';

const HomeScreen = ({ navigation }) => {
  return (
    <View style={styles.container}>
      {/* Your existing buttons */}
      
      {/* ADD THIS NEW BUTTON */}
      <TouchableOpacity 
        style={styles.whoAmIButton}
        onPress={() => navigation.navigate('WhoIsThis')}
      >
        <Text style={styles.whoAmIButtonText}>👤 Who Am I?</Text>
      </TouchableOpacity>
      
      {/* Optional: Add family management buttons */}
      <TouchableOpacity 
        style={styles.addFamilyButton}
        onPress={() => navigation.navigate('AddFamilyMember')}
      >
        <Text style={styles.buttonText}>➕ Add Family Member</Text>
      </TouchableOpacity>
      
      <TouchableOpacity 
        style={styles.familyListButton}
        onPress={() => navigation.navigate('FamilyList')}
      >
        <Text style={styles.buttonText}>📋 View Family List</Text>
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
    backgroundColor: '#f5f5f5',
  },
  whoAmIButton: {
    backgroundColor: '#007AFF',
    paddingVertical: 20,
    paddingHorizontal: 40,
    borderRadius: 15,
    marginVertical: 10,
    minWidth: 250,
    alignItems: 'center',
    // Make it prominent for dementia patients
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 5,
    elevation: 8,
  },
  whoAmIButtonText: {
    color: 'white',
    fontSize: 24,
    fontWeight: 'bold',
  },
  addFamilyButton: {
    backgroundColor: '#34C759',
    paddingVertical: 15,
    paddingHorizontal: 30,
    borderRadius: 12,
    marginVertical: 5,
    minWidth: 250,
    alignItems: 'center',
  },
  familyListButton: {
    backgroundColor: '#FF9500',
    paddingVertical: 15,
    paddingHorizontal: 30,
    borderRadius: 12,
    marginVertical: 5,
    minWidth: 250,
    alignItems: 'center',
  },
  buttonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
  },
});

/**
 * STEP 4: File Structure in Your React Native Project
 */

// Add these files to your project:
// 
// src/services/
//   └── FamilyRecognitionAPI.js  (copy from generated file)
// 
// src/screens/
//   ├── AddFamilyMemberScreen.js  (copy from generated file)
//   ├── WhoIsThisScreen.js        (copy from generated file)
//   └── FamilyListScreen.js       (copy from generated file)

/**
 * STEP 5: Update app.json (for Expo projects)
 */

// Add camera permissions to your app.json:
{
  "expo": {
    "name": "Your App Name",
    "permissions": [
      "CAMERA",
      "CAMERA_ROLL"
    ],
    "ios": {
      "infoPlist": {
        "NSCameraUsageDescription": "This app needs camera access to recognize family members for dementia patients."
      }
    },
    "android": {
      "permissions": [
        "android.permission.CAMERA",
        "android.permission.WRITE_EXTERNAL_STORAGE"
      ]
    }
  }
}

/**
 * STEP 6: Test Backend Connection
 */

// Add this test function to verify backend is working:
import { FamilyRecognitionAPI } from '../services/FamilyRecognitionAPI';

const testBackend = async () => {
  try {
    const status = await FamilyRecognitionAPI.checkBackendStatus();
    if (status) {
      console.log('✅ Backend connected successfully!');
      Alert.alert('Success', 'Backend is connected and ready to use!');
    } else {
      Alert.alert('Error', 'Cannot connect to backend. Make sure server is running on 10.200.253.165:8000');
    }
  } catch (error) {
    Alert.alert('Error', `Backend connection failed: ${error.message}`);
  }
};

// Call testBackend() when app starts to verify connection

/**
 * STEP 7: Start Your Backend Server
 */

// In your Python project directory, run:
// python -m uvicorn app_backend:app --host 0.0.0.0 --port 8000 --reload

/**
 * STEP 8: Usage Flow for Dementia Patients
 */

// 1. Patient opens app and sees "Who Am I?" button (large, prominent)
// 2. Patient taps button → Camera opens with simple instructions
// 3. Patient takes photo → App shows "This is [Name]" immediately
// 4. If person not recognized → Option to "Add This Person"
// 5. Adding person captures 50 photos automatically with progress shown
// 6. Family members can be managed in the Family List screen

/**
 * STEP 9: Key Features Implemented
 */

// ✅ Large, senior-friendly interface
// ✅ Instant face recognition with clear results
// ✅ Automatic 50-photo capture for training
// ✅ Family member management (add/view/delete)
// ✅ Error handling and offline graceful degradation
// ✅ Progress indicators and user feedback
// ✅ Simple navigation optimized for dementia patients

/**
 * STEP 10: Troubleshooting
 */

// Common issues and solutions:

// 1. "Cannot connect to backend"
//    → Check if backend server is running: http://10.200.253.165:8000
//    → Ensure phone and computer are on same WiFi network
//    → Check Windows Firewall allows port 8000

// 2. "Camera permission denied"
//    → Update app.json with camera permissions
//    → Request permissions in app settings

// 3. "Person not recognized"
//    → Make sure 50 photos were captured during training
//    → Try adding person again with better lighting
//    → Check backend logs for recognition errors

// 4. "422 Unprocessable Entity"
//    → Backend API validation issue
//    → Check that photos are being sent in correct format
//    → Verify FormData structure matches backend expectations

/**
 * READY TO USE!
 * 
 * All code files are generated with your IP address (10.200.253.165).
 * Just copy the files to your React Native project and add the navigation routes.
 * The backend should be running on your computer for the app to work.
 */

export default HomeScreen;