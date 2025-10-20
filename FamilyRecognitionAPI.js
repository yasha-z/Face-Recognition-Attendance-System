// Family Recognition API Service for React Native
// Ready to use with your IP: 10.200.253.165

const BASE_URL = 'http://10.200.253.165:8000';

export class FamilyRecognitionAPI {
  
  // Check if backend is running
  static async checkBackendStatus() {
    try {
      const response = await fetch(`${BASE_URL}/api/status`);
      const status = await response.json();
      console.log(`Backend ready. ${status.total_family_members} family members registered.`);
      return status;
    } catch (error) {
      console.log('Backend not available:', error.message);
      return null;
    }
  }

  // Add new family member with 50 photos
  static async addFamilyMember(name, relationship, photos) {
    const formData = new FormData();
    formData.append('name', name);
    formData.append('relationship', relationship);
    
    // Add all photos to FormData
    photos.forEach((photo, index) => {
      formData.append('images', {
        uri: photo.uri,
        type: 'image/jpeg',
        name: `${name}_${index}.jpg`,
      });
    });

    try {
      const response = await fetch(`${BASE_URL}/api/family-members`, {
        method: 'POST',
        body: formData,
        // Don't set Content-Type header - let it be automatic for FormData
      });
      
      const result = await response.json();
      
      if (response.ok) {
        console.log(`✅ ${name} added successfully! Training model...`);
        return result;
      } else {
        throw new Error(result.detail || 'Failed to add family member');
      }
    } catch (error) {
      console.error('❌ Add family member error:', error);
      throw error;
    }
  }

  // Identify person from photo
  static async identifyPerson(photoUri) {
    const formData = new FormData();
    formData.append('image', {
      uri: photoUri,
      type: 'image/jpeg',
      name: 'identify.jpg',
    });

    try {
      const response = await fetch(`${BASE_URL}/api/recognition/identify`, {
        method: 'POST',
        body: formData,
      });
      
      const result = await response.json();
      
      if (response.ok) {
        console.log(`🔍 Recognition result:`, result);
        return result; // { person_name: "John", status: "identified" }
      } else {
        console.log('⚠️ Recognition failed:', result.detail);
        return { person_name: 'Unknown', status: 'not_recognized' };
      }
    } catch (error) {
      console.error('❌ Recognition error:', error);
      return { person_name: 'Unknown', status: 'error' };
    }
  }

  // Get all family members
  static async getFamilyMembers() {
    try {
      const response = await fetch(`${BASE_URL}/api/family-members`);
      const result = await response.json();
      
      if (response.ok) {
        console.log(`📋 Found ${result.family_members.length} family members`);
        return result.family_members;
      } else {
        throw new Error('Failed to fetch family members');
      }
    } catch (error) {
      console.error('❌ Error fetching family members:', error);
      return [];
    }
  }

  // Delete family member
  static async deleteFamilyMember(memberId) {
    try {
      const response = await fetch(`${BASE_URL}/api/family-members/${memberId}`, {
        method: 'DELETE',
      });
      
      const result = await response.json();
      
      if (response.ok) {
        console.log(`🗑️ Deleted family member: ${memberId}`);
        return result;
      } else {
        throw new Error(result.detail || 'Failed to delete family member');
      }
    } catch (error) {
      console.error('❌ Delete family member error:', error);
      throw error;
    }
  }

  // Start live recognition (if needed)
  static async startLiveRecognition() {
    try {
      const response = await fetch(`${BASE_URL}/api/recognition/start`, {
        method: 'POST',
      });
      
      return await response.json();
    } catch (error) {
      console.error('❌ Start live recognition error:', error);
      throw error;
    }
  }
}

// Example usage:
/*
import { FamilyRecognitionAPI } from './FamilyRecognitionAPI';

// Check backend
const status = await FamilyRecognitionAPI.checkBackendStatus();

// Add family member
const photos = [...]; // Array of 50 captured photos
await FamilyRecognitionAPI.addFamilyMember('John', 'son', photos);

// Identify person
const result = await FamilyRecognitionAPI.identifyPerson(photoUri);
console.log(`This is: ${result.person_name}`);

// Get all family members
const familyMembers = await FamilyRecognitionAPI.getFamilyMembers();
*/