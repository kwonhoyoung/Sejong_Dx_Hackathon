// app/(tabs)/report.tsx
import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, ScrollView, Alert, Pressable, Linking } from 'react-native';
import { Stack, useLocalSearchParams } from 'expo-router';
import HeaderLeftGoBack from '../../components/HeaderLeftGoBack'; 

// SearchResponse 인터페이스: 백엔드에서 받는 JSON 데이터의 구조 정의
interface SearchResponse {
  제목: string;
  태그: string[];
  보고서: {
    '정리된 내용': string; // 띄어쓰기 포함된 키
    'AI가 제공하는 리포트': string; // 띄어쓰기 포함된 키
    '출처 링크': string[]; // 띄어쓰기 포함된 키
  };
}

export default function ReportScreen() {
  const params = useLocalSearchParams();
  const [reportData, setReportData] = useState<SearchResponse | null>(null);
  const [showSources, setShowSources] = useState(false);

  useEffect(() => {
    if (params.searchData) {
      try {
        const parsedData: SearchResponse = JSON.parse(params.searchData as string);
        setReportData(parsedData);
        
        console.log("ReportScreen: 제목 (원본):", parsedData.제목);
        console.log("ReportScreen: AI가 제공하는 리포트 (원본):", parsedData.보고서['AI가 제공하는 리포트']);
        console.log("ReportScreen: 출처 링크 (원본):", parsedData.보고서['출처 링크']);

      } catch (error) {
        console.error('ReportScreen: 데이터 파싱 오류:', error);
        Alert.alert('오류', '검색 결과를 불러오는 중 오류가 발생했습니다.');
        setReportData(null);
      }
    } else {
      Alert.alert('알림', '검색 결과 데이터가 없습니다.');
      setReportData(null);
    }
  }, [params.searchData]);

  // ⭐ 강화된 cleanMarkdownString 헬퍼 함수
  const cleanMarkdownString = (text: string | undefined): string => {
    if (!text) return '';
    let cleaned = text.trim();

    // 1. 선행하는 ```json 제거 (제목과 리포트 모두 해당)
    cleaned = cleaned.replace(/^```json\s*/, '');
    // 2. 후행하는 ``` 제거
    cleaned = cleaned.replace(/\s*```$/, '');

    // 3. AI가 제공하는 리포트 내용에 특화된 추가 정리
    // - "주요 인사이트:\n1. {" 또는 "주요 인사이트:\n1. {" 같은 패턴 제거
    cleaned = cleaned.replace(/주요 인사이트:\s*\d+\.\s*\{/, '주요 인사이트:\n');
    // - "trend_summary": "..." 패턴에서 키와 따옴표 제거
    cleaned = cleaned.replace(/"trend_summary":\s*"/g, '');
    // - ", "insights": [" 패턴에서 키와 따옴표, 괄호 제거
    cleaned = cleaned.replace(/",\s*"insights":\s*\[/g, '\n\n인사이트:\n');
    // - "future_outlook": "..." 패턴에서 키와 따옴표 제거
    cleaned = cleaned.replace(/"future_outlook":\s*"/g, '\n\n향후 전망: ');
    // - 남은 모든 큰따옴표 제거
    cleaned = cleaned.replace(/"/g, ''); 
    // - 마지막에 남은 불완전한 JSON 괄호/대괄호/쉼표 제거
    cleaned = cleaned.replace(/\[\s*$/, ''); 
    cleaned = cleaned.replace(/\]\s*$/, ''); 
    cleaned = cleaned.replace(/,\s*$/, ''); 
    cleaned = cleaned.replace(/\{\s*$/, ''); 
    cleaned = cleaned.replace(/\}\s*$/, ''); 
    // - 줄 시작의 숫자. 공백 (예: "1. ", "2. ") 제거 (multiline 모드)
    cleaned = cleaned.replace(/^\d+\.\s*/gm, ''); 
    
    // 최종적으로 앞뒤 공백 제거 및 여러 줄 공백을 한 줄로 줄이기
    cleaned = cleaned.trim().replace(/\n\s*\n/g, '\n\n'); 

    return cleaned;
  };

  const openLink = async (url: string) => {
    try {
      const supported = await Linking.canOpenURL(url);
      if (supported) {
        await Linking.openURL(url);
      } else {
        Alert.alert('오류', `이 링크를 열 수 없습니다: ${url}`);
      }
    } catch (error) {
      console.error('링크 열기 오류:', error);
      Alert.alert('오류', '링크를 여는 중 오류가 발생했습니다.');
    }
  };

  return (
    <View style={styles.container}>
      <Stack.Screen
        options={{
          headerShown: true,
          // ⭐ 상단 중앙에 제목 표시 (cleanMarkdownString 적용)
          headerTitle: reportData ? cleanMarkdownString(reportData.제목) : '', 
          headerTitleStyle: styles.headerTitle,
          // ⭐ HeaderLeftGoBack에 "홈 | [제목]" 표시 (cleanMarkdownString 적용)
          headerLeft: () => (
            <HeaderLeftGoBack 
              title={reportData ? cleanMarkdownString(reportData.제목) : '이전'} 
            />
          ), 
          headerBackground: () => <View style={{ flex: 1, backgroundColor: '#f8f9fa' }} />,
          headerShadowVisible: false,
        }}
      />
      
      <ScrollView style={styles.scrollView} contentContainerStyle={styles.scrollContent}>
        {reportData ? (
          <>

            {/* `정리된 내용` 필드에 띄어쓰기가 있으므로 대괄호 표기법 사용 */}
            {reportData.보고서['정리된 내용'] && ( 
              <>
                <Text style={styles.sectionTitle}>정리된 내용</Text>
                <Text style={styles.content}>{reportData.보고서['정리된 내용']}</Text>
              </>
            )}

            {/* `AI가 제공하는 리포트` 필드에 띄어쓰기가 있으므로 대괄호 표기법 사용 */}
            {reportData.보고서['AI가 제공하는 리포트'] && (
              <>
                <Text style={styles.sectionTitle}>AI가 제공하는 리포트</Text>
                {/* AI가 제공하는 리포트 내용에 cleanMarkdownString 적용 */}
                <Text style={styles.content}>{cleanMarkdownString(reportData.보고서['AI가 제공하는 리포트'])}</Text>
              </>
            )}


            <View style={styles.tagsContainer}>
              {reportData.태그.map((tag, index) => (
                <View key={index} style={styles.tag}>
                  <Text style={styles.tagText}>{tag}</Text>
                </View>
              ))}
            </View>

            <Pressable 
              style={styles.sourceButton} 
              onPress={() => setShowSources(!showSources)}
            >
              <Text style={styles.sourceButtonText}>
                {showSources ? '출처 닫기' : '출처 보기'}
              </Text>
            </Pressable>
            
            {/* `출처 링크` 필드에 띄어쓰기가 있으므로 대괄호 표기법 사용 */}
            {showSources && reportData.보고서['출처 링크'] && reportData.보고서['출처 링크'].length > 0 && ( 
              <View style={styles.sourcesList}>
                {reportData.보고서['출처 링크'].map((link, index) => (
                  <Pressable key={index} onPress={() => openLink(link)}>
                    <Text style={styles.sourceLink}>{link}</Text>
                  </Pressable>
                ))}
              </View>
            )}
            {showSources && (!reportData.보고서['출처 링크'] || reportData.보고서['출처 링크'].length === 0) && (
                <Text style={styles.noSourceText}>제공된 출처 링크가 없습니다.</Text>
            )}
          </>
        ) : (
          <Text style={styles.loadingText}>보고서 데이터를 불러오는 중...</Text>
        )}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  scrollView: {
    flex: 1,
    padding: 15,
  },
  scrollContent: {
    paddingBottom: 20,
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    flexShrink: 1,
  },
  tagsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginBottom: 15,
  },
  tag: {
    backgroundColor: '#e0e0e0',
    borderRadius: 5,
    paddingVertical: 5,
    paddingHorizontal: 10,
    marginRight: 8,
    marginBottom: 8,
  },
  tagText: {
    fontSize: 12,
    color: '#555',
    fontWeight: 'bold',
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginTop: 20,
    marginBottom: 10,
    color: '#333',
  },
  content: {
    fontSize: 15,
    lineHeight: 22,
    color: '#444',
    marginBottom: 15,
  },
  sourceButton: {
    backgroundColor: '#FBCEB1',
    paddingVertical: 10,
    paddingHorizontal: 15,
    borderRadius: 8,
    alignSelf: 'flex-start',
    marginTop: 20,
    marginBottom: 10,
  },
  sourceButtonText: {
    color: '#000',
    fontSize: 15,
    fontWeight: 'bold',
  },
  sourcesList: {
    marginTop: 10,
    padding: 10,
    backgroundColor: '#f9f9f9',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#eee',
  },
  sourceLink: {
    fontSize: 14,
    color: '#007AFF',
    textDecorationLine: 'underline',
    marginBottom: 10, 
  },
  noSourceText: {
    fontSize: 14,
    color: '#888',
    fontStyle: 'italic',
    marginTop: 10,
  },
  loadingText: {
    textAlign: 'center',
    marginTop: 50,
    fontSize: 16,
    color: '#666',
  },
});