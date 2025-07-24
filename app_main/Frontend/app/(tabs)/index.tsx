import { Image } from 'expo-image';
import { Platform, StyleSheet, ScrollView, ActivityIndicator, TouchableOpacity, Alert } from 'react-native';
import { View, Text, TextInput } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useState } from 'react';

// 백엔드 서버 URL 설정 (개발 환경에 맞게 수정하세요)
const API_URL = Platform.OS === 'ios'
  ? 'http://localhost:8000'
  : 'http://10.0.2.2:8000'; // Android Emulator의 경우

interface SearchResult {
  제목: string;
  태그: string[];
  보고서: {
    정리된_내용: string;
    AI가_제공하는_리포트: string;
    출처_링크: string[];
  };
}

export default function HomeScreen() {
    const [search, setSearch] = useState('');
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState<SearchResult | null>(null);
    const [error, setError] = useState<string | null>(null);

    const handleSearch = async () => {
        if (!search.trim()) {
            Alert.alert('알림', '검색어를 입력해주세요.');
            return;
        }

        setLoading(true);
        setError(null);
        setResult(null);

        try {
            const response = await fetch(`${API_URL}/api/search`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    topic: search,
                    time_period: '최근 1주일',
                    analyze: true
                }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            if (data.error) {
                setError(data.error);
            } else {
                setResult(data);
            }
        } catch (err) {
            console.error('Search error:', err);
            setError('검색 중 오류가 발생했습니다. 네트워크 연결을 확인해주세요.');
        } finally {
            setLoading(false);
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
                    onSubmitEditing={handleSearch}
                    returnKeyType="search"
                    editable={!loading}
                />
                <TouchableOpacity
                    onPress={handleSearch}
                    disabled={loading}
                    style={styles.searchButton}
                >
                    {loading ? (
                        <ActivityIndicator size="small" color="#fff" />
                    ) : (
                        <Text style={styles.searchButtonText}>검색</Text>
                    )}
                </TouchableOpacity>
            </View>

            <ScrollView style={styles.resultContainer} showsVerticalScrollIndicator={false}>
                {error && (
                    <View style={styles.errorContainer}>
                        <Text style={styles.errorText}>{error}</Text>
                    </View>
                )}

                {result && (
                    <View style={styles.resultContent}>
                        <Text style={styles.resultTitle}>{result.제목}</Text>

                        <View style={styles.tagContainer}>
                            {result.태그.map((tag, index) => (
                                <View key={index} style={styles.tag}>
                                    <Text style={styles.tagText}>{tag}</Text>
                                </View>
                            ))}
                        </View>

                        <View style={styles.section}>
                            <Text style={styles.sectionTitle}>정리된 내용</Text>
                            <Text style={styles.sectionContent}>{result.보고서.정리된_내용}</Text>
                        </View>

                        <View style={styles.section}>
                            <Text style={styles.sectionTitle}>AI 분석 리포트</Text>
                            <Text style={styles.sectionContent}>{result.보고서.AI가_제공하는_리포트}</Text>
                        </View>

                        {result.보고서.출처_링크.length > 0 && (
                            <View style={styles.section}>
                                <Text style={styles.sectionTitle}>출처</Text>
                                {result.보고서.출처_링크.map((link, index) => (
                                    <Text key={index} style={styles.linkText}>• {link}</Text>
                                ))}
                            </View>
                        )}
                    </View>
                )}
            </ScrollView>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        paddingTop: 100,
        alignItems: 'center',
        backgroundColor: '#fff',
    },
    searchContainer: {
        flexDirection: 'row',
        alignItems: 'center',
        width: '90%',
        height: 50,
        borderWidth: 0,
        paddingHorizontal: 15,
        borderRadius: 50,
        backgroundColor: '#FBCEB1',
        fontSize: 18,
    },
    searchIcon: {
        marginRight: 10,
    },
    input: {
        flex: 1,
        fontSize: 18,
    },
    searchButton: {
        backgroundColor: '#FF8C42',
        paddingHorizontal: 20,
        paddingVertical: 10,
        borderRadius: 25,
        marginLeft: 10,
    },
    searchButtonText: {
        color: '#fff',
        fontWeight: 'bold',
        fontSize: 16,
    },
    resultContainer: {
        flex: 1,
        width: '100%',
        marginTop: 20,
        paddingHorizontal: 20,
    },
    errorContainer: {
        backgroundColor: '#FFE6E6',
        padding: 15,
        borderRadius: 10,
        marginTop: 20,
    },
    errorText: {
        color: '#D32F2F',
        fontSize: 16,
        textAlign: 'center',
    },
    resultContent: {
        paddingBottom: 50,
    },
    resultTitle: {
        fontSize: 24,
        fontWeight: 'bold',
        marginTop: 20,
        marginBottom: 15,
        color: '#333',
    },
    tagContainer: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        marginBottom: 20,
    },
    tag: {
        backgroundColor: '#E3F2FD',
        paddingHorizontal: 15,
        paddingVertical: 8,
        borderRadius: 20,
        marginRight: 10,
        marginBottom: 10,
    },
    tagText: {
        color: '#1976D2',
        fontSize: 14,
        fontWeight: '500',
    },
    section: {
        marginBottom: 25,
    },
    sectionTitle: {
        fontSize: 20,
        fontWeight: 'bold',
        marginBottom: 10,
        color: '#444',
    },
    sectionContent: {
        fontSize: 16,
        lineHeight: 24,
        color: '#666',
    },
    linkText: {
        fontSize: 14,
        color: '#1976D2',
        marginBottom: 5,
        textDecorationLine: 'underline',
    },