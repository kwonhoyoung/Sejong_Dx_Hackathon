import React, { useState } from 'react';
import { View, Text, TextInput, StyleSheet, Alert, ActivityIndicator, Pressable } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';

interface SearchResponse {
  제목: string;
  태그: string[];
  보고서: {
    정리된내용: string;
    AI가제공하는리포트: string;
    출처링크: string[];
  };
}

export default function HomeScreen() {
  const [search, setSearch] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const router = useRouter();

  // 검색 함수
  const handleSearch = async () => {
    if (!search.trim()) {
      Alert.alert('검색어 입력', '검색어를 입력해주세요.');
      return;
    }

    setIsLoading(true);
    
    try {
      // 백엔드 API 호출 (안드로이드 에뮬레이터용)
      const response = await fetch('http://10.0.2.2:8000/api/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          topic: search,
          time_period: '최근 1주일',
          analyze: true
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: SearchResponse = await response.json();
      
      // 검색 결과를 report 페이지로 전달
      // 데이터를 query parameter로 전달하는 대신 전역 상태나 AsyncStorage 사용 권장
      // 여기서는 간단한 예시를 위해 navigation state로 전달
      router.push({
        pathname: '/report',
        params: { searchData: JSON.stringify(data) }
      });

    } catch (error) {
      console.error('검색 오류:', error);
      Alert.alert(
        '검색 실패', 
        '검색 중 오류가 발생했습니다. 네트워크 연결을 확인해주세요.'
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      <View style={styles.searchContainer}>
        <Ionicons name='search' size={24} color='black' style={styles.searchIcon}/>
        <TextInput
          placeholder='Search 검색'
          style={styles.input}
          value={search}
          onChangeText={setSearch}
          onSubmitEditing={handleSearch} // 엔터키로도 검색 가능
          returnKeyType="search"
          editable={!isLoading}
        />
        
        {/* 검색 버튼 */}
        <Pressable 
          style={[styles.searchButton, isLoading && styles.searchButtonDisabled]}
          onPress={handleSearch}
          disabled={isLoading}
        >
          {isLoading ? (
            <ActivityIndicator size="small" color="#fff" />
          ) : (
            <Text style={styles.searchButtonText}>검색</Text>
          )}
        </Pressable>
      </View>

      {/* 로딩 상태 표시 */}
      {isLoading && (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#007AFF" />
          <Text style={styles.loadingText}>검색 중...</Text>
        </View>
      )}

    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,      
    paddingTop: 300,
    alignItems: 'center',
    backgroundColor: '#fff',
    paddingHorizontal: 20,
  },
  searchContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    width: '100%',
    height: 50,
    borderWidth: 0,
    paddingHorizontal: 15,
    borderRadius: 50,
    backgroundColor: '#e5e5e5',
    fontSize: 18,
    marginBottom: 30,
  },
  searchIcon: {
    position: 'absolute',
    marginLeft: 15,
  },
  input: {
    flex: 1,
    fontSize: 20,
    marginLeft: 30,
    marginRight: 10,
  },
  searchButton: {
    backgroundColor: '#FBCEB1',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    minWidth: 60,
    alignItems: 'center',
    justifyContent: 'center',
  },
  searchButtonDisabled: {
    backgroundColor: '#ccc',
  },
  searchButtonText: {
    color: '#000',
    fontSize: 14,
    fontWeight: 'bold',
  },
  loadingContainer: {
    alignItems: 'center',
    marginTop: 50,
  },
  loadingText: {
    marginTop: 10,
    fontSize: 16,
    color: '#666',
  },
  instructionContainer: {
    alignItems: 'center',
    marginTop: 50,
    padding: 20,
    backgroundColor: '#f8f9fa',
    borderRadius: 15,
    width: '100%',
  },
  instructionTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 15,
    color: '#333',
  },
  instructionText: {
    fontSize: 16,
    textAlign: 'center',
    lineHeight: 24,
    color: '#666',
    marginBottom: 15,
  },
  exampleText: {
    fontSize: 14,
    color: '#999',
    fontStyle: 'italic',
    textAlign: 'center',
  },
});