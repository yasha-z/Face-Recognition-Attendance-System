// AddFamilyMemberScreen.js
// Screen for adding new family members with 50 photo capture

import React, { useState, useRef, useEffect } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  Alert,
  SafeAreaView,
  ActivityIndicator,
} from 'react-native';
import { Camera } from 'expo-camera';
import { Picker } from '@react-native-picker/picker';
import { FamilyRecognitionAPI } from './FamilyRecognitionAPI';

const AddFamilyMemberScreen = ({ navigation }) => {
  const [name, setName] = useState('');
  const [relationship, setRelationship] = useState('son');
  const [hasPermission, setHasPermission] = useState(null);
  const [isCapturing, setIsCapturing] = useState(false);
  const [currentPhoto, setCurrentPhoto] = useState(0);
  const [capturedPhotos, setCapturedPhotos] = useState([]);
  const [isTraining, setIsTraining] = useState(false);
  const cameraRef = useRef(null);

  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === 'granted');
    })();
  }, []);

  const relationships = [
    'spouse', 'child', 'parent', 'sibling', 
    'grandchild', 'friend', 'caregiver', 'other'
  ];

  const startPhotoCapture = async () => {
    if (!name.trim()) {
      Alert.alert('Error', 'Please enter a name first');
      return;
    }

    setIsCapturing(true);
    setCurrentPhoto(0);
    setCapturedPhotos([]);

    try {
      const photos = [];
      
      for (let i = 0; i < 50; i++) {
        setCurrentPhoto(i + 1);
        
        const photo = await cameraRef.current.takePictureAsync({
          quality: 0.8,
          base64: false,
        });
        
        photos.push(photo);
        
        // Small delay between captures (like Flask app)
        await new Promise(resolve => setTimeout(resolve, 200));
      }
      
      setCapturedPhotos(photos);
      setIsCapturing(false);
      
      // Automatically start training
      await trainModel(photos);
      
    } catch (error) {
      console.error('Photo capture error:', error);
      Alert.alert('Error', 'Failed to capture photos');
      setIsCapturing(false);
    }
  };

  const trainModel = async (photos) => {
    setIsTraining(true);
    
    try {
      const result = await FamilyRecognitionAPI.addFamilyMember(
        name.trim(), 
        relationship, 
        photos
      );
      
      Alert.alert(
        'Success!', 
        `${name} has been added successfully!\n\nModel training complete.`,
        [
          {
            text: 'OK',
            onPress: () => navigation.goBack(),
          },
        ]
      );
      
    } catch (error) {
      Alert.alert('Error', `Failed to add family member: ${error.message}`);
    } finally {
      setIsTraining(false);
    }
  };

  if (hasPermission === null) {
    return <View style={styles.container}><Text>Requesting camera permission...</Text></View>;
  }
  
  if (hasPermission === false) {
    return (
      <View style={styles.container}>
        <Text style={styles.errorText}>No access to camera</Text>
      </View>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      {!isCapturing && !isTraining && (
        <View style={styles.formContainer}>
          <Text style={styles.title}>Add Family Member</Text>
          
          <Text style={styles.label}>Name:</Text>
          <TextInput
            style={styles.input}
            value={name}
            onChangeText={setName}
            placeholder="Enter family member's name"
            placeholderTextColor="#999"
          />
          
          <Text style={styles.label}>Relationship:</Text>
          <View style={styles.pickerContainer}>
            <Picker
              selectedValue={relationship}
              onValueChange={setRelationship}
              style={styles.picker}
            >
              {relationships.map((rel) => (
                <Picker.Item key={rel} label={rel.charAt(0).toUpperCase() + rel.slice(1)} value={rel} />
              ))}
            </Picker>
          </View>
          
          <TouchableOpacity 
            style={styles.startButton} 
            onPress={startPhotoCapture}
            disabled={!name.trim()}
          >
            <Text style={styles.buttonText}>Start Photo Capture</Text>
          </TouchableOpacity>
        </View>
      )}

      {(isCapturing || isTraining) && (
        <View style={styles.cameraContainer}>
          <Camera 
            style={styles.camera} 
            ref={cameraRef}
            type={Camera.Constants.Type.front}
          />
          
          <View style={styles.overlay}>
            <Text style={styles.nameText}>Adding: {name}</Text>
            
            {isCapturing && (
              <>
                <Text style={styles.progressText}>Photo {currentPhoto}/50</Text>
                <View style={styles.progressBar}>
                  <View 
                    style={[
                      styles.progressFill, 
                      { width: `${(currentPhoto / 50) * 100}%` }
                    ]} 
                  />
                </View>
                <Text style={styles.instruction}>Look at the camera</Text>
              </>
            )}
            
            {isTraining && (
              <>
                <ActivityIndicator size="large" color="#ffffff" />
                <Text style={styles.trainingText}>Training AI model...</Text>
                <Text style={styles.instructionSmall}>This may take a moment</Text>
              </>
            )}
          </View>
        </View>
      )}
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  formContainer: {
    flex: 1,
    padding: 20,
    justifyContent: 'center',
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 30,
    color: '#333',
  },
  label: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 10,
    color: '#333',
  },
  input: {
    height: 60,
    borderColor: '#ddd',
    borderWidth: 2,
    borderRadius: 10,
    paddingHorizontal: 15,
    fontSize: 18,
    marginBottom: 20,
    backgroundColor: '#fff',
  },
  pickerContainer: {
    borderColor: '#ddd',
    borderWidth: 2,
    borderRadius: 10,
    marginBottom: 30,
    backgroundColor: '#fff',
  },
  picker: {
    height: 60,
    fontSize: 18,
  },
  startButton: {
    backgroundColor: '#007AFF',
    padding: 20,
    borderRadius: 15,
    alignItems: 'center',
  },
  buttonText: {
    color: 'white',
    fontSize: 20,
    fontWeight: 'bold',
  },
  cameraContainer: {
    flex: 1,
  },
  camera: {
    flex: 1,
  },
  overlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.3)',
  },
  nameText: {
    fontSize: 24,
    fontWeight: 'bold',
    color: 'white',
    marginBottom: 20,
  },
  progressText: {
    fontSize: 32,
    fontWeight: 'bold',
    color: 'white',
    marginBottom: 20,
  },
  progressBar: {
    width: '80%',
    height: 8,
    backgroundColor: 'rgba(255,255,255,0.3)',
    borderRadius: 4,
    marginBottom: 20,
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#4CAF50',
    borderRadius: 4,
  },
  instruction: {
    fontSize: 20,
    color: 'white',
    textAlign: 'center',
  },
  trainingText: {
    fontSize: 24,
    fontWeight: 'bold',
    color: 'white',
    marginTop: 20,
  },
  instructionSmall: {
    fontSize: 16,
    color: 'white',
    marginTop: 10,
  },
  errorText: {
    fontSize: 18,
    color: 'red',
    textAlign: 'center',
  },
});

export default AddFamilyMemberScreen;