import React, { useState, useEffect } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  Image,
  ScrollView,
  Alert,
  ActivityIndicator,
  SafeAreaView,
  Dimensions,
  Animated,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import * as FileSystem from 'expo-file-system';
import { LinearGradient } from 'expo-linear-gradient';
import * as Network from 'expo-network';

const { width, height } = Dimensions.get('window');

// Configure your server URL here - UPDATE THIS WITH YOUR ACTUAL RENDER URL!
const SERVER_URL = 'https://cattle-disease-api.onrender.com'; // ⚠️ CHANGE THIS!

interface PredictionResult {
  class: string;
  stage?: string;
  description: string;
  confidence: number;
}

export default function HomeScreen() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [serverStatus, setServerStatus] = useState<'checking' | 'connected' | 'disconnected' | 'error'>('checking');
  
  // Animation values
  const fadeAnim = new Animated.Value(0);
  const scaleAnim = new Animated.Value(0.95);
  const slideAnim = new Animated.Value(30);

  useEffect(() => {
    checkServerHealth();
    
    // Keep-alive mechanism - ping server every 10 minutes
    const keepAliveInterval = setInterval(() => {
      if (serverStatus === 'connected') {
        fetch(`${SERVER_URL}/health`).catch(() => {}); // Silent ping
      }
    }, 600000); // 10 minutes
    
    // Entrance animation
    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 800,
        useNativeDriver: true,
      }),
      Animated.timing(scaleAnim, {
        toValue: 1,
        duration: 600,
        useNativeDriver: true,
      }),
      Animated.timing(slideAnim, {
        toValue: 0,
        duration: 800,
        useNativeDriver: true,
      }),
    ]).start();

    // Cleanup interval on unmount
    return () => clearInterval(keepAliveInterval);
  }, []);

  const checkServerHealth = async (): Promise<void> => {
    try {
      setServerStatus('checking');
      console.log('Checking server health at:', SERVER_URL);
      
      // Add timeout for wake-up scenarios
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 45000); // 45 second timeout
      
      const response = await fetch(`${SERVER_URL}/health`, {
        signal: controller.signal,
        method: 'GET',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
        }
      });
      
      clearTimeout(timeoutId);
      
      console.log('Response status:', response.status);
      console.log('Response headers:', response.headers);
      
      // Log the raw response text first
      const responseText = await response.text();
      console.log('Raw response:', responseText);
      
      // Try to parse as JSON
      let data;
      try {
        data = JSON.parse(responseText);
      } catch (parseError) {
        console.error('JSON Parse Error:', parseError);
        console.error('Response was:', responseText.substring(0, 500)); // Show first 500 chars
        setServerStatus('error');
        Alert.alert(
          'Server Error', 
          `Server returned invalid response. Check console for details.\n\nResponse: ${responseText.substring(0, 100)}...`
        );
        return;
      }
      
      if (data && data.status === 'healthy') {
        setServerStatus('connected');
        console.log('Server is healthy:', data);
      } else {
        setServerStatus('error');
        console.log('Server response:', data);
      }
    } catch (error) {
      console.error('Server health check failed:', error);
      if (error.name === 'AbortError') {
        setServerStatus('disconnected');
        Alert.alert('Server Timeout', 'The server is taking too long to respond. It might be sleeping or starting up.');
      } else {
        setServerStatus('disconnected');
        Alert.alert('Connection Error', `Cannot reach server: ${error.message}`);
      }
    }
  };

  const pickImage = async (source: 'camera' | 'gallery'): Promise<void> => {
    try {
      let result: ImagePicker.ImagePickerResult;
      
      if (source === 'camera') {
        const permission = await ImagePicker.requestCameraPermissionsAsync();
        if (!permission.granted) {
          Alert.alert('Permission needed', 'Camera permission is required');
          return;
        }
        result = await ImagePicker.launchCameraAsync({
          mediaTypes: ImagePicker.MediaTypeOptions.Images,
          allowsEditing: true,
          aspect: [1, 1],
          quality: 0.8,
        });
      } else {
        const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
        if (!permission.granted) {
          Alert.alert('Permission needed', 'Gallery permission is required');
          return;
        }
        result = await ImagePicker.launchImageLibraryAsync({
          mediaTypes: ImagePicker.MediaTypeOptions.Images,
          allowsEditing: true,
          aspect: [1, 1],
          quality: 0.8,
        });
      }

      if (!result.canceled && result.assets && result.assets[0]) {
        setSelectedImage(result.assets[0].uri);
        setPrediction(null);
      }
    } catch (error) {
      console.error('Error picking image:', error);
      Alert.alert('Error', 'Failed to pick image');
    }
  };

  const uploadAndClassify = async (): Promise<void> => {
    if (!selectedImage) {
      Alert.alert('Error', 'Please select an image first');
      return;
    }

    // Check network connectivity first
    try {
      const networkState = await Network.getNetworkStateAsync();
      if (!networkState.isConnected) {
        Alert.alert('No Internet', 'Please check your internet connection and try again.');
        return;
      }
    } catch (error) {
      console.log('Network check failed:', error);
    }

    if (serverStatus !== 'connected') {
      Alert.alert('Server Issue', 'Server is not connected. Checking status...');
      await checkServerHealth();
      return;
    }

    setLoading(true);
    
    try {
      console.log('Starting image classification...');
      console.log('Server URL:', SERVER_URL);
      
      // Show wake-up message for longer requests
      const startTime = Date.now();
      
      // Convert image to base64
      console.log('Converting image to base64...');
      const base64Image = await FileSystem.readAsStringAsync(selectedImage, {
        encoding: FileSystem.EncodingType.Base64,
      });
      
      console.log('Base64 conversion complete. Length:', base64Image.length);

      // Prepare the request
      const requestBody = {
        image: base64Image
      };

      console.log('Sending request to server...');
      
      // Send to server with better error handling
      const response = await fetch(`${SERVER_URL}/classify`, {
        method: 'POST',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
        // Note: timeout is not a standard fetch option, you may need to use AbortController
      });

      console.log('Response received. Status:', response.status);
      console.log('Response headers:', response.headers);
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('Server error response:', errorText);
        throw new Error(`Server error: ${response.status} - ${errorText}`);
      }

      const responseText = await response.text();
      console.log('Raw response:', responseText);
      
      let data;
      try {
        data = JSON.parse(responseText);
      } catch (parseError) {
        console.error('JSON parse error:', parseError);
        console.error('Response text:', responseText);
        throw new Error(`Invalid JSON response: ${responseText.substring(0, 100)}`);
      }

      const elapsed = Date.now() - startTime;
      console.log('Request completed in:', elapsed, 'ms');
      
      if (elapsed > 10000) {
        Alert.alert('Server Was Sleeping', 'The server was asleep and has now woken up. Future requests will be faster!');
      }

      if (data.success) {
        console.log('Classification successful:', data.prediction);
        setPrediction(data.prediction);
      } else {
        throw new Error(data.error || 'Classification failed');
      }

    } catch (error) {
      console.error('Error classifying image:', error);
      
      // Better error messages based on error type
      if (error.message.includes('Network request failed')) {
        Alert.alert(
          'Network Error', 
          'Cannot connect to server. Please check:\n\n' +
          '• Your internet connection\n' +
          '• Server URL is correct\n' +
          '• Server is running\n\n' +
          `Current server: ${SERVER_URL}`
        );
      } else if (error.message.includes('timeout')) {
        Alert.alert('Request Timeout', 'The server is taking too long to respond. It might be sleeping. Please try again in a moment.');
      } else if (error.message.includes('JSON')) {
        Alert.alert('Server Error', 'Server returned invalid response. Please check server logs.');
      } else {
        Alert.alert('Error', `Classification failed: ${error.message}`);
      }
    } finally {
      setLoading(false);
    }
  };

  const getServerStatusIcon = (): string => {
    switch (serverStatus) {
      case 'connected': return '🟢';
      case 'disconnected': return '🔴';
      case 'error': return '🟡';
      default: return '⚪';
    }
  };

  const getConfidenceColor = (confidence: number): string => {
    if (confidence > 0.8) return '#00C853';
    if (confidence > 0.6) return '#FF9500';
    return '#FF3B30';
  };

  return (
    <SafeAreaView style={styles.container}>
      <LinearGradient
        colors={['#667eea', '#764ba2']}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={styles.gradient}
      >
        <ScrollView 
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
        >
          {/* Premium Header */}
          <Animated.View 
            style={[
              styles.header,
              {
                opacity: fadeAnim,
                transform: [{ translateY: slideAnim }]
              }
            ]}
          >
            <View style={styles.headerContent}>
              <View style={styles.titleContainer}>
                <Text style={styles.mainTitle}>VetAI</Text>
                <Text style={styles.subtitle}>Cattle Disease Detection</Text>
              </View>
              
              <View style={styles.statusContainer}>
                <View style={[
                  styles.statusBadge,
                  { 
                    backgroundColor: serverStatus === 'connected' ? '#00C853' : 
                                   serverStatus === 'checking' ? '#FF9500' : '#FF3B30'
                  }
                ]}>
                  <Text style={styles.statusText}>
                    {serverStatus === 'connected' ? 'ONLINE' : 
                     serverStatus === 'checking' ? 'WAKING...' : 'OFFLINE'}
                  </Text>
                </View>
              </View>
            </View>
          </Animated.View>

          {/* Main Content */}
          <Animated.View 
            style={[
              styles.mainContent,
              {
                opacity: fadeAnim,
                transform: [{ scale: scaleAnim }]
              }
            ]}
          >
            {/* Debug Section - Remove in production */}
            <View style={styles.debugSection}>
              <TouchableOpacity 
                style={styles.debugButton}
                onPress={() => {
                  Alert.alert('Debug Info', `Server URL: ${SERVER_URL}\nStatus: ${serverStatus}`);
                }}
              >
                <Text style={styles.debugText}>Debug Info</Text>
              </TouchableOpacity>
              
              <TouchableOpacity 
                style={styles.debugButton}
                onPress={async () => {
                  try {
                    const response = await fetch(`${SERVER_URL}/`);
                    const text = await response.text();
                    Alert.alert('Server Response', text.substring(0, 200));
                  } catch (error) {
                    Alert.alert('Error', error.message);
                  }
                }}
              >
                <Text style={styles.debugText}>Test Server</Text>
              </TouchableOpacity>
            </View>

            {/* Action Buttons */}
            <View style={styles.buttonContainer}>
              <TouchableOpacity 
                style={styles.actionButton} 
                onPress={() => pickImage('camera')}
                activeOpacity={0.9}
              >
                <LinearGradient
                  colors={['#FF6B35', '#F7931E']}
                  style={styles.buttonGradient}
                >
                  <View style={styles.buttonContent}>
                    <View style={styles.iconCircle}>
                      <Text style={styles.buttonIcon}>📸</Text>
                    </View>
                    <Text style={styles.buttonTitle}>Camera</Text>
                    <Text style={styles.buttonSubtitle}>Take Photo</Text>
                  </View>
                </LinearGradient>
              </TouchableOpacity>
              
              <TouchableOpacity 
                style={styles.actionButton} 
                onPress={() => pickImage('gallery')}
                activeOpacity={0.9}
              >
                <LinearGradient
                  colors={['#4ECDC4', '#44A08D']}
                  style={styles.buttonGradient}
                >
                  <View style={styles.buttonContent}>
                    <View style={styles.iconCircle}>
                      <Text style={styles.buttonIcon}>🖼️</Text>
                    </View>
                    <Text style={styles.buttonTitle}>Gallery</Text>
                    <Text style={styles.buttonSubtitle}>Choose Image</Text>
                  </View>
                </LinearGradient>
              </TouchableOpacity>
            </View>

            {/* Selected Image Section */}
            {selectedImage && (
              <Animated.View style={styles.imageSection}>
                <View style={styles.imageCard}>
                  <View style={styles.imageContainer}>
                    <Image source={{ uri: selectedImage }} style={styles.selectedImage} />
                    <LinearGradient
                      colors={['transparent', 'rgba(0,0,0,0.4)']}
                      style={styles.imageOverlay}
                    />
                  </View>
                  
                  <TouchableOpacity 
                    style={[
                      styles.analyzeButton,
                      serverStatus !== 'connected' && styles.disabledButton
                    ]} 
                    onPress={uploadAndClassify}
                    disabled={loading || serverStatus !== 'connected'}
                    activeOpacity={0.9}
                  >
                    <LinearGradient
                      colors={loading ? ['#9E9E9E', '#757575'] : ['#667eea', '#764ba2']}
                      style={styles.analyzeGradient}
                    >
                      {loading ? (
                        <View style={styles.loadingContainer}>
                          <ActivityIndicator color="white" size="small" />
                          <Text style={styles.loadingText}>Analyzing...</Text>
                        </View>
                      ) : (
                        <View style={styles.analyzeContent}>
                          <Text style={styles.analyzeIcon}>🔬</Text>
                          <Text style={styles.analyzeText}>Analyze Image</Text>
                        </View>
                      )}
                    </LinearGradient>
                  </TouchableOpacity>
                </View>
              </Animated.View>
            )}

            {/* Results Section */}
            {prediction && !loading && (
              <Animated.View style={styles.resultsSection}>
                <View style={styles.resultsCard}>
                  <LinearGradient
                    colors={['#ffffff', '#f8f9fa']}
                    style={styles.resultsGradient}
                  >
                    <View style={styles.resultsHeader}>
                      <View style={styles.resultsIconContainer}>
                        <Text style={styles.resultsIcon}>🎯</Text>
                      </View>
                      <Text style={styles.resultsTitle}>Diagnosis Results</Text>
                    </View>
                    
                    {/* Disease Information */}
                    <View style={styles.diseaseSection}>
                      <LinearGradient
                        colors={['#FFE5E5', '#FFF0F0']}
                        style={styles.infoCard}
                      >
                        <View style={styles.cardHeader}>
                          <View style={styles.cardIconContainer}>
                            <Text style={styles.cardIcon}>🦠</Text>
                          </View>
                          <Text style={styles.cardTitle}>Disease Detected</Text>
                        </View>
                        <Text style={styles.diseaseName}>{prediction.class}</Text>
                        
                        {prediction.stage && prediction.stage.trim() && (
                          <View style={styles.stageContainer}>
                            <Text style={styles.stageLabel}>Stage:</Text>
                            <Text style={styles.stageName}>{prediction.stage}</Text>
                          </View>
                        )}
                      </LinearGradient>
                    </View>
                    
                    {/* Confidence Level */}
                    <View style={styles.confidenceSection}>
                      <LinearGradient
                        colors={['#E8F5E8', '#F0F8F0']}
                        style={styles.infoCard}
                      >
                        <View style={styles.cardHeader}>
                          <View style={styles.cardIconContainer}>
                            <Text style={styles.cardIcon}>📊</Text>
                          </View>
                          <Text style={styles.cardTitle}>Confidence Level</Text>
                        </View>
                        
                        <View style={styles.confidenceDisplay}>
                          <Text 
                            style={[
                              styles.confidenceValue,
                              { color: getConfidenceColor(prediction.confidence) }
                            ]}
                          >
                            {(prediction.confidence * 100).toFixed(1)}%
                          </Text>
                          
                          <View style={styles.confidenceBarContainer}>
                            <View style={styles.confidenceBar}>
                              <LinearGradient
                                colors={[getConfidenceColor(prediction.confidence), getConfidenceColor(prediction.confidence) + '80']}
                                style={[
                                  styles.confidenceFill,
                                  { width: `${prediction.confidence * 100}%` }
                                ]}
                              />
                            </View>
                          </View>
                        </View>
                      </LinearGradient>
                    </View>
                    
                    {/* Description */}
                    <View style={styles.descriptionSection}>
                      <LinearGradient
                        colors={['#E3F2FD', '#F1F8FF']}
                        style={styles.infoCard}
                      >
                        <View style={styles.cardHeader}>
                          <View style={styles.cardIconContainer}>
                            <Text style={styles.cardIcon}>📝</Text>
                          </View>
                          <Text style={styles.cardTitle}>Clinical Description</Text>
                        </View>
                        <Text style={styles.descriptionText}>{prediction.description}</Text>
                      </LinearGradient>
                    </View>
                  </LinearGradient>
                </View>
              </Animated.View>
            )}

            {/* Empty State */}
            {!selectedImage && (
              <View style={styles.emptyState}>
                <LinearGradient
                  colors={['rgba(255,255,255,0.95)', 'rgba(255,255,255,0.85)']}
                  style={styles.emptyCard}
                >
                  <View style={styles.emptyIconContainer}>
                    <Text style={styles.emptyIcon}>🐄</Text>
                  </View>
                  <Text style={styles.emptyTitle}>Ready for Diagnosis</Text>
                  <Text style={styles.emptySubtitle}>
                    Select an image using the camera or gallery to begin AI-powered disease detection
                  </Text>
                </LinearGradient>
              </View>
            )}
          </Animated.View>
        </ScrollView>
      </LinearGradient>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  gradient: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
  },
  header: {
    paddingTop: 20,
    paddingBottom: 30,
    paddingHorizontal: 20,
  },
  headerContent: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  titleContainer: {
    flex: 1,
  },
  mainTitle: {
    fontSize: 36,
    fontWeight: '800',
    color: 'white',
    letterSpacing: -1,
    textShadowColor: 'rgba(0,0,0,0.3)',
    textShadowOffset: { width: 0, height: 2 },
    textShadowRadius: 8,
  },
  subtitle: {
    fontSize: 16,
    color: 'rgba(255,255,255,0.9)',
    fontWeight: '400',
    marginTop: 4,
    letterSpacing: 0.5,
  },
  statusContainer: {
    alignItems: 'flex-end',
  },
  statusBadge: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 4,
  },
  statusText: {
    color: 'white',
    fontSize: 12,
    fontWeight: '700',
    letterSpacing: 1,
  },
  mainContent: {
    flex: 1,
    paddingHorizontal: 20,
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 30,
  },
  actionButton: {
    flex: 0.48,
    borderRadius: 20,
    elevation: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
  },
  buttonGradient: {
    borderRadius: 20,
    padding: 20,
    minHeight: 120,
  },
  buttonContent: {
    alignItems: 'center',
    flex: 1,
    justifyContent: 'center',
  },
  iconCircle: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: 'rgba(255,255,255,0.2)',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 12,
  },
  buttonIcon: {
    fontSize: 24,
  },
  buttonTitle: {
    color: 'white',
    fontSize: 18,
    fontWeight: '700',
    marginBottom: 4,
  },
  buttonSubtitle: {
    color: 'rgba(255,255,255,0.8)',
    fontSize: 13,
    fontWeight: '500',
  },
  imageSection: {
    marginBottom: 20,
  },
  imageCard: {
    backgroundColor: 'white',
    borderRadius: 20,
    padding: 15,
    elevation: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 5 },
    shadowOpacity: 0.2,
    shadowRadius: 10,
  },
  imageContainer: {
    borderRadius: 15,
    overflow: 'hidden',
    marginBottom: 15,
  },
  selectedImage: {
    width: '100%',
    height: 250,
    resizeMode: 'cover',
  },
  imageOverlay: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    height: 50,
  },
  analyzeButton: {
    borderRadius: 15,
    overflow: 'hidden',
  },
  analyzeGradient: {
    paddingVertical: 16,
    alignItems: 'center',
  },
  loadingContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  loadingText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 10,
  },
  analyzeContent: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  analyzeIcon: {
    fontSize: 20,
    marginRight: 10,
  },
  analyzeText: {
    color: 'white',
    fontSize: 18,
    fontWeight: '700',
  },
  disabledButton: {
    opacity: 0.6,
  },
  resultsSection: {
    marginBottom: 20,
  },
  resultsCard: {
    borderRadius: 20,
    overflow: 'hidden',
    elevation: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.2,
    shadowRadius: 12,
  },
  resultsGradient: {
    padding: 25,
  },
  resultsHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 25,
    paddingBottom: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#E0E0E0',
  },
  resultsIconContainer: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: '#667eea',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 15,
  },
  resultsIcon: {
    fontSize: 24,
  },
  resultsTitle: {
    fontSize: 24,
    fontWeight: '700',
    color: '#2D3748',
  },
  diseaseSection: {
    marginBottom: 20,
  },
  confidenceSection: {
    marginBottom: 20,
  },
  descriptionSection: {
    marginBottom: 0,
  },
  infoCard: {
    borderRadius: 15,
    padding: 20,
    elevation: 3,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  cardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 15,
  },
  cardIconContainer: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(255,255,255,0.8)',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  cardIcon: {
    fontSize: 20,
  },
  cardTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#2D3748',
    letterSpacing: 0.5,
  },
  diseaseName: {
    fontSize: 22,
    fontWeight: '800',
    color: '#2D3748',
    marginBottom: 10,
  },
  stageContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  stageLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#718096',
    marginRight: 8,
  },
  stageName: {
    fontSize: 16,
    fontWeight: '700',
    color: '#4A5568',
  },
  confidenceDisplay: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  confidenceValue: {
    fontSize: 28,
    fontWeight: '800',
    marginRight: 20,
    minWidth: 90,
  },
  confidenceBarContainer: {
    flex: 1,
  },
  confidenceBar: {
    height: 12,
    backgroundColor: '#E2E8F0',
    borderRadius: 6,
    overflow: 'hidden',
  },
  confidenceFill: {
    height: '100%',
    borderRadius: 6,
  },
  descriptionText: {
    fontSize: 16,
    color: '#4A5568',
    lineHeight: 24,
    fontWeight: '500',
  },
  emptyState: {
    marginTop: 40,
    alignItems: 'center',
  },
  emptyCard: {
    width: '100%',
    paddingVertical: 50,
    paddingHorizontal: 30,
    borderRadius: 20,
    alignItems: 'center',
    elevation: 6,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.1,
    shadowRadius: 6,
  },
  emptyIconContainer: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: 'rgba(102, 126, 234, 0.1)',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 20,
  },
  emptyIcon: {
    fontSize: 40,
  },
  emptyTitle: {
    fontSize: 24,
    fontWeight: '700',
    color: '#2D3748',
    marginBottom: 10,
    textAlign: 'center',
  },
  emptySubtitle: {
    fontSize: 16,
    color: '#718096',
    textAlign: 'center',
    lineHeight: 22,
    fontWeight: '500',
  },
  debugSection: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: 20,
    padding: 10,
    backgroundColor: 'rgba(255,255,255,0.1)',
    borderRadius: 10,
  },
  debugButton: {
    backgroundColor: 'rgba(255,255,255,0.2)',
    padding: 10,
    borderRadius: 8,
    minWidth: 100,
    alignItems: 'center',
  },
  debugText: {
    color: 'white',
    fontSize: 12,
    fontWeight: '600',
  },
});
