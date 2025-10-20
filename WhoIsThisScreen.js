// WhoIsThisScreen.js
// Main recognition screen - "Who am I?" feature

import React, { useState, useRef, useEffect } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Alert,
  SafeAreaView,
  ActivityIndicator,
} from 'react-native';
import { Camera } from 'expo-camera';
import { FamilyRecognitionAPI } from './FamilyRecognitionAPI';

const WhoIsThisScreen = ({ navigation }) => {
  const [hasPermission, setHasPermission] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [showCamera, setShowCamera] = useState(true);
  const cameraRef = useRef(null);

  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === 'granted');
    })();
  }, []);

  const takePhoto = async () => {
    setIsAnalyzing(true);
    setResult(null);

    try {
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.8,
        base64: false,
      });

      // Call recognition API
      const recognitionResult = await FamilyRecognitionAPI.identifyPerson(photo.uri);
      
      setResult(recognitionResult);
      setShowCamera(false);
      
    } catch (error) {
      console.error('Recognition error:', error);
      Alert.alert('Error', 'Failed to analyze photo. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const tryAgain = () => {
    setResult(null);
    setShowCamera(true);
  };

  const addThisPerson = () => {
    // Navigate to add family member screen
    navigation.navigate('AddFamilyMember');
  };

  if (hasPermission === null) {
    return (
      <View style={styles.container}>
        <Text style={styles.loadingText}>Requesting camera permission...</Text>
      </View>
    );
  }
  
  if (hasPermission === false) {
    return (
      <View style={styles.container}>
        <Text style={styles.errorText}>No access to camera</Text>
        <Text style={styles.errorSubtext}>Please enable camera permissions in settings</Text>
      </View>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      {showCamera && (
        <View style={styles.cameraContainer}>
          <Camera 
            style={styles.camera} 
            ref={cameraRef}
            type={Camera.Constants.Type.front}
          />
          
          <View style={styles.overlay}>
            <Text style={styles.title}>Who Am I?</Text>
            <Text style={styles.instruction}>
              Point the camera at your face and tap the button
            </Text>
            
            <TouchableOpacity 
              style={[styles.captureButton, isAnalyzing && styles.disabledButton]} 
              onPress={takePhoto}
              disabled={isAnalyzing}
            >
              {isAnalyzing ? (
                <>
                  <ActivityIndicator size="large" color="#ffffff" />
                  <Text style={styles.buttonText}>Analyzing...</Text>
                </>
              ) : (
                <Text style={styles.buttonText}>Take Photo</Text>
              )}
            </TouchableOpacity>
          </View>
        </View>
      )}

      {result && !showCamera && (
        <View style={styles.resultsContainer}>
          {result.status === 'identified' ? (
            <>
              <Text style={styles.successTitle}>This is:</Text>
              <Text style={styles.personName}>{result.person_name}</Text>
              
              <TouchableOpacity style={styles.tryAgainButton} onPress={tryAgain}>
                <Text style={styles.tryAgainText}>Try Again</Text>
              </TouchableOpacity>
              
              <TouchableOpacity 
                style={styles.backButton} 
                onPress={() => navigation.goBack()}
              >
                <Text style={styles.backButtonText}>Back to Menu</Text>
              </TouchableOpacity>
            </>
          ) : (
            <>
              <Text style={styles.errorTitle}>Person not recognized</Text>
              <Text style={styles.errorMessage}>
                I don't recognize this person yet.
              </Text>
              
              <TouchableOpacity style={styles.addPersonButton} onPress={addThisPerson}>
                <Text style={styles.addPersonText}>Add This Person</Text>
              </TouchableOpacity>
              
              <TouchableOpacity style={styles.tryAgainButton} onPress={tryAgain}>
                <Text style={styles.tryAgainText}>Try Again</Text>
              </TouchableOpacity>
            </>
          )}
        </View>
      )}
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
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
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 60,
    backgroundColor: 'rgba(0,0,0,0.3)',
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    color: 'white',
    textAlign: 'center',
  },
  instruction: {
    fontSize: 20,
    color: 'white',
    textAlign: 'center',
    paddingHorizontal: 20,
    marginBottom: 20,
  },
  captureButton: {
    backgroundColor: '#007AFF',
    paddingVertical: 20,
    paddingHorizontal: 40,
    borderRadius: 50,
    alignItems: 'center',
    minWidth: 200,
  },
  disabledButton: {
    backgroundColor: '#666',
  },
  buttonText: {
    color: 'white',
    fontSize: 22,
    fontWeight: 'bold',
    marginTop: 5,
  },
  resultsContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f5f5f5',
    padding: 20,
  },
  successTitle: {
    fontSize: 24,
    color: '#333',
    marginBottom: 10,
  },
  personName: {
    fontSize: 48,
    fontWeight: 'bold',
    color: '#007AFF',
    textAlign: 'center',
    marginBottom: 40,
  },
  errorTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#FF3B30',
    textAlign: 'center',
    marginBottom: 20,
  },
  errorMessage: {
    fontSize: 20,
    color: '#666',
    textAlign: 'center',
    marginBottom: 40,
    paddingHorizontal: 20,
  },
  tryAgainButton: {
    backgroundColor: '#34C759',
    paddingVertical: 15,
    paddingHorizontal: 30,
    borderRadius: 15,
    marginBottom: 20,
    minWidth: 200,
    alignItems: 'center',
  },
  tryAgainText: {
    color: 'white',
    fontSize: 20,
    fontWeight: 'bold',
  },
  addPersonButton: {
    backgroundColor: '#FF9500',
    paddingVertical: 15,
    paddingHorizontal: 30,
    borderRadius: 15,
    marginBottom: 20,
    minWidth: 200,
    alignItems: 'center',
  },
  addPersonText: {
    color: 'white',
    fontSize: 20,
    fontWeight: 'bold',
  },
  backButton: {
    backgroundColor: '#8E8E93',
    paddingVertical: 15,
    paddingHorizontal: 30,
    borderRadius: 15,
    minWidth: 200,
    alignItems: 'center',
  },
  backButtonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
  },
  loadingText: {
    fontSize: 18,
    color: 'white',
    textAlign: 'center',
    marginTop: 100,
  },
  errorText: {
    fontSize: 20,
    color: '#FF3B30',
    textAlign: 'center',
    marginTop: 100,
  },
  errorSubtext: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
    marginTop: 20,
    paddingHorizontal: 40,
  },
});

export default WhoIsThisScreen;