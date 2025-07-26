// app/(tabs)/report.tsx
import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, ScrollView, Alert, Pressable, Linking } from 'react-native';
import { Stack, useLocalSearchParams } from 'expo-router';
// HeaderLeftGoBack 컴포넌트 경로 확인. 이 경로는 'app/(tabs)/report.tsx' 기준입니다.
import HeaderLeftGoBack from '../../components/HeaderLeftGoBack'; 

// SearchResponse 인터페이스: HomeScreen과 동일하게 구조 정의 (키 이름 띄어쓰기 포함)
interface SearchResponse {
  제목: string;
  태그: string[];
  보고서: {
    '정리된 내용': string; 
    'AI가 제공하는 리포트': string; 
    '출처 링크': string[]; 
  };
}

export default function ReportScreen() {
  const params = useLocalSearchParams(); // URL 파라미터 가져오기
  const [reportData, setReportData] = useState<SearchResponse | null>(null); // 보고서 데이터 상태
  const [showSources, setShowSources] = useState(false); // 출처 표시 여부 상태 (토글용)

  // 컴포넌트 마운트 시 또는 params.searchData 변경 시 데이터 파싱
  useEffect(() => {
    if (params.searchData) {
      try {
        const parsedData: SearchResponse = JSON.parse(params.searchData as string);
        setReportData(parsedData);
        // 디버깅을 위한 콘솔 로그 (데이터가 제대로 파싱되었는지 확인)
        console.log("ReportScreen: 제목:", parsedData.제목);
        console.log("ReportScreen: 태그:", parsedData.태그);
        console.log("ReportScreen: 정리된 내용:", parsedData.보고서['정리된 내용'] ? "존재함" : "없음");
        console.log("ReportScreen: AI가 제공하는 리포트:", parsedData.보고서['AI가 제공하는 리포트'] ? "존재함" : "없음");
        console.log("ReportScreen: 출처 링크:", parsedData.보고서['출처 링크'] ? "존재함" : "없음");

      } catch (error) {
        console.error('ReportScreen: 데이터 파싱 오류:', error);
        Alert.alert('오류', '검색 결과를 불러오는 중 오류가 발생했습니다.');
        setReportData(null);
      }
    } else {
      Alert.alert('알림', '검색 결과 데이터가 없습니다.');
      setReportData(null);
    }
  }, [params.searchData]); // params.searchData가 변경될 때마다 실행

  // 출처 링크를 외부 브라우저로 열기 위한 함수
  const openLink = async (url: string) => {
    try {
      const supported = await Linking.canOpenURL(url); // 해당 URL을 열 수 있는지 확인
      if (supported) {
        await Linking.openURL(url); // URL 열기
      } else {
        Alert.alert('오류', `이 링크를 열 수 없습니다: ${url}`); // 열 수 없는 경우 알림
      }
    } catch (error) {
      console.error('링크 열기 오류:', error);
      Alert.alert('오류', '링크를 여는 중 오류가 발생했습니다.');
    }
  };

  return (
    <View style={styles.container}>
      {/* Expo Router 스택 스크린 헤더 설정 */}
      <Stack.Screen
        options={{
          headerShown: true, // 헤더 표시 여부
          // 헤더 타이틀을 동적으로 보고서 제목으로 설정
          headerTitle: reportData ? reportData.제목 : '보고서', 
          headerTitleStyle: styles.headerTitle, // 헤더 타이틀 스타일 적용
          // 사용자 정의 뒤로가기 버튼 컴포넌트
          headerLeft: () => <HeaderLeftGoBack title="이전" />, 
          headerBackground: () => <View style={{ flex: 1, backgroundColor: '#f8f9fa' }} />, // 헤더 배경색
          headerShadowVisible: false, // 헤더 그림자 비활성화
        }}
      />
      
      {/* 보고서 내용을 스크롤 가능하게 하는 ScrollView */}
      <ScrollView style={styles.scrollView} contentContainerStyle={styles.scrollContent}>
        {reportData ? ( // reportData가 존재할 때만 내용 렌더링
          <>
            {/* 태그 섹션 */}
            <View style={styles.tagsContainer}>
              {reportData.태그.map((tag, index) => (
                <View key={index} style={styles.tag}>
                  <Text style={styles.tagText}>{tag}</Text>
                </View>
              ))}
            </View>

            {/* 정리된 내용 섹션 */}
            {/* 데이터가 있을 경우에만 섹션 표시 */}
            {reportData.보고서['정리된 내용'] && ( 
              <>
                <Text style={styles.sectionTitle}>정리된 내용</Text>
                <Text style={styles.content}>{reportData.보고서['정리된 내용']}</Text>
              </>
            )}

            {/* AI가 제공하는 리포트 섹션 */}
            {/* 데이터가 있을 경우에만 섹션 표시 */}
            {reportData.보고서['AI가 제공하는 리포트'] && (
              <>
                <Text style={styles.sectionTitle}>AI가 제공하는 리포트</Text>
                <Text style={styles.content}>{reportData.보고서['AI가 제공하는 리포트']}</Text>
              </>
            )}

            {/* 출처 버튼 - 클릭 시 출처 목록 토글 */}
            <Pressable 
              style={styles.sourceButton} 
              onPress={() => setShowSources(!showSources)}
            >
              <Text style={styles.sourceButtonText}>
                {showSources ? '출처 닫기' : '출처 보기'}
              </Text>
            </Pressable>

            {/* 출처 목록 - showSources가 true이고 출처 링크가 있을 경우에만 표시 */}
            {showSources && reportData.보고서['출처 링크'] && reportData.보고서['출처 링크'].length > 0 && (
              <View style={styles.sourcesList}>
                {reportData.보고서['출처 링크'].map((link, index) => (
                  <Pressable key={index} onPress={() => openLink(link)}>
                    <Text style={styles.sourceLink}>{link}</Text>
                  </Pressable>
                ))}
              </View>
            )}
            {/* 출처는 보고 싶은데 출처 링크가 없는 경우 메시지 표시 */}
            {showSources && (!reportData.보고서['출처 링크'] || reportData.보고서['출처 링크'].length === 0) && (
                <Text style={styles.noSourceText}>제공된 출처 링크가 없습니다.</Text>
            )}
          </>
        ) : (
          // reportData가 아직 없거나 로딩 중일 때 표시할 메시지
          <Text style={styles.loadingText}>보고서 데이터를 불러오는 중...</Text>
        )}
      </ScrollView>
    </View>
  );
}

// 컴포넌트 스타일 정의
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  scrollView: {
    flex: 1,
    padding: 15, // ScrollView 내부 전체 패딩
  },
  scrollContent: {
    paddingBottom: 20, // ScrollView 하단 여백
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    flexShrink: 1, // 텍스트가 길어질 경우 줄어들도록
  },
  tagsContainer: {
    flexDirection: 'row', // 태그를 가로로 나열
    flexWrap: 'wrap', // 공간 부족 시 다음 줄로 넘김
    marginBottom: 15,
  },
  tag: {
    backgroundColor: '#e0e0e0', // 태그 배경색
    borderRadius: 5, // 둥근 사각형 모양
    paddingVertical: 5,
    paddingHorizontal: 10,
    marginRight: 8, // 태그 간 간격
    marginBottom: 8, // 태그 줄 간 간격
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
    backgroundColor: '#FBCEB1', // 버튼 색상
    paddingVertical: 10,
    paddingHorizontal: 15,
    borderRadius: 8,
    alignSelf: 'flex-start', // 내용에 맞춰 버튼 너비 조절
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
    color: '#007AFF', // 링크 색상
    textDecorationLine: 'underline', // 밑줄
    marginBottom: 5,
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