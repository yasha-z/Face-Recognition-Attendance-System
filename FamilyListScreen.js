// FamilyListScreen.js
// Screen to view and manage all family members

import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  FlatList,
  TouchableOpacity,
  StyleSheet,
  Alert,
  SafeAreaView,
  ActivityIndicator,
  RefreshControl,
} from 'react-native';
import { FamilyRecognitionAPI } from './FamilyRecognitionAPI';

const FamilyListScreen = ({ navigation }) => {
  const [familyMembers, setFamilyMembers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    loadFamilyMembers();
  }, []);

  const loadFamilyMembers = async () => {
    try {
      const members = await FamilyRecognitionAPI.getFamilyMembers();
      setFamilyMembers(members);
    } catch (error) {
      Alert.alert('Error', 'Failed to load family members');
    } finally {
      setLoading(false);
    }
  };

  const onRefresh = async () => {
    setRefreshing(true);
    await loadFamilyMembers();
    setRefreshing(false);
  };

  const deleteMember = async (member) => {
    Alert.alert(
      'Delete Family Member',
      `Are you sure you want to remove ${member.name}?`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            try {
              await FamilyRecognitionAPI.deleteFamilyMember(member.id);
              await loadFamilyMembers(); // Refresh list
              Alert.alert('Success', `${member.name} has been removed`);
            } catch (error) {
              Alert.alert('Error', `Failed to delete ${member.name}`);
            }
          },
        },
      ]
    );
  };

  const renderFamilyMember = ({ item }) => (
    <View style={styles.memberCard}>
      <View style={styles.memberInfo}>
        <Text style={styles.memberName}>{item.name}</Text>
        <Text style={styles.memberRelationship}>
          {item.relationship.charAt(0).toUpperCase() + item.relationship.slice(1)}
        </Text>
        <Text style={styles.memberDate}>Added: {item.date_added}</Text>
      </View>
      
      <TouchableOpacity 
        style={styles.deleteButton}
        onPress={() => deleteMember(item)}
      >
        <Text style={styles.deleteButtonText}>Remove</Text>
      </TouchableOpacity>
    </View>
  );

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#007AFF" />
        <Text style={styles.loadingText}>Loading family members...</Text>
      </View>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Family Members</Text>
        <Text style={styles.count}>
          {familyMembers.length} {familyMembers.length === 1 ? 'person' : 'people'} registered
        </Text>
      </View>

      {familyMembers.length === 0 ? (
        <View style={styles.emptyContainer}>
          <Text style={styles.emptyText}>No family members added yet</Text>
          <Text style={styles.emptySubtext}>
            Add family members so I can recognize them
          </Text>
          
          <TouchableOpacity 
            style={styles.addFirstButton}
            onPress={() => navigation.navigate('AddFamilyMember')}
          >
            <Text style={styles.addFirstButtonText}>Add First Family Member</Text>
          </TouchableOpacity>
        </View>
      ) : (
        <FlatList
          data={familyMembers}
          keyExtractor={(item) => item.id}
          renderItem={renderFamilyMember}
          contentContainerStyle={styles.listContainer}
          refreshControl={
            <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
          }
        />
      )}

      {familyMembers.length > 0 && (
        <TouchableOpacity 
          style={styles.addMoreButton}
          onPress={() => navigation.navigate('AddFamilyMember')}
        >
          <Text style={styles.addMoreButtonText}>+ Add Another Family Member</Text>
        </TouchableOpacity>
      )}
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f5f5f5',
  },
  loadingText: {
    fontSize: 18,
    color: '#666',
    marginTop: 20,
  },
  header: {
    padding: 20,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#333',
    textAlign: 'center',
  },
  count: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
    marginTop: 5,
  },
  listContainer: {
    padding: 20,
  },
  memberCard: {
    backgroundColor: '#fff',
    borderRadius: 15,
    padding: 20,
    marginBottom: 15,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  memberInfo: {
    flex: 1,
  },
  memberName: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 5,
  },
  memberRelationship: {
    fontSize: 18,
    color: '#007AFF',
    marginBottom: 5,
  },
  memberDate: {
    fontSize: 14,
    color: '#666',
  },
  deleteButton: {
    backgroundColor: '#FF3B30',
    paddingVertical: 10,
    paddingHorizontal: 15,
    borderRadius: 8,
  },
  deleteButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 40,
  },
  emptyText: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
    textAlign: 'center',
    marginBottom: 10,
  },
  emptySubtext: {
    fontSize: 18,
    color: '#666',
    textAlign: 'center',
    marginBottom: 40,
  },
  addFirstButton: {
    backgroundColor: '#007AFF',
    paddingVertical: 20,
    paddingHorizontal: 30,
    borderRadius: 15,
  },
  addFirstButtonText: {
    color: 'white',
    fontSize: 20,
    fontWeight: 'bold',
  },
  addMoreButton: {
    backgroundColor: '#34C759',
    margin: 20,
    paddingVertical: 15,
    borderRadius: 12,
    alignItems: 'center',
  },
  addMoreButtonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
  },
});

export default FamilyListScreen;