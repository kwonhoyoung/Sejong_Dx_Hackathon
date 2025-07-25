import React, { useState, useEffect } from 'react';
import { View, StyleSheet, FlatList, Text, Alert } from 'react-native';
import { Stack, Link, useLocalSearchParams } from 'expo-router';
import ReportItem from '../../components/ReportItem';
import HeaderLeftGoBack from '../../components/HeaderLeftGoBack';

interface ReportData {
  id: string;
  title: string;
  tags: string[];
  content?: string;
  aiReport?: string;
  sources?: string[];
}

interface SearchResponse {
  제목: string;
  태그: string[];
  보고서: {
    정리된내용: string;
    AI가제공하는리포트: string;
    출처링크: string[];
  };
}

// 백엔드 API가 완성되기 전 사용할 임시 데이터
const MOCK_DATA: ReportData[] = [
  { id: '1', title: 'AI 리버싱 기술, 코드게이트 2025 핵심 화두 부상', tags: ['리버싱', 'AI리버싱', '코드게이트2025'] },
  { id: '2', title: '클라우드 보안의 미래, 서버리스 아키텍처의 역할', tags: ['클라우드', '보안', '서버리스'] },
  { id: '3', title: 'LLM을 활용한 효과적인 챗봇 개발 전략', tags: ['LLM', '챗봇', '자연어처리'] },
  { id: '4', title: '데이터 프라이버시, GDPR 준수를 위한 기술적 접근', tags: ['데이터', '프라이버시', 'GDPR'] },
  { id: '5', title: '양자 컴퓨팅이 암호화 기술에 미치는 영향 분석', tags: ['양자컴퓨팅', '암호화', '보안'] },
];

export default function ReportScreen() {
  const [reports, setReports] = useState<ReportData[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const params = useLocalSearchParams();

  // 검색 결과를 ReportData 형식으로 변환하는 함수
  const convertSearchResponseToReportData = (searchResponse: SearchResponse): ReportData => {
    return {
      id: Date.now().toString(), // 임시 ID 생성
      title: searchResponse.제목,
      tags: searchResponse.태그,
      content: searchResponse.보고서.정리된내용,
      aiReport: searchResponse.보고서.AI가제공하는리포트,
      sources: searchResponse.보고서.출처링크,
    };
  };

  useEffect(() => {
    // 검색 결과가 있는 경우에만 처리
    if (params.searchData) {
      try {
        const searchResponse: SearchResponse = JSON.parse(params.searchData as string);
        const reportData = convertSearchResponseToReportData(searchResponse);
        setReports([reportData]); // 검색 결과만 설정
      } catch (error) {
        console.error('검색 데이터 파싱 오류:', error);
        Alert.alert('오류', '검색 결과를 불러오는 중 오류가 발생했습니다.');
        setReports([]); // 오류 시 빈 배열로 설정
      }
    } else {
      // 검색 결과가 없는 경우 빈 배열로 설정
      setReports([]);
    }
  }, [params.searchData]);

  // 추가 검색 기능 - 현재는 비활성화 (필요시 활성화)
  const loadMoreReports = async () => {
    // 실제 구현에서는 더 많은 검색 결과나 다른 주제의 보고서를 불러옴
    // 현재는 검색 결과만 표시하므로 이 기능은 사용하지 않음
    return;
  };  

  const renderFooter = () => {
    if (isLoading) {
      return (
        <View style={styles.footerLoader}>
          <Text>더 많은 보고서를 불러오는 중...</Text>
        </View>
      );
    }
    return null;
  };

  const renderEmptyComponent = () => (
    <View style={styles.emptyContainer}>
      <Text style={styles.emptyTitle}>🔍 검색 결과가 없습니다</Text>
      <Text style={styles.emptyText}>
        홈 화면에서 원하는 주제를 검색해보세요.{'\n'}
        AI가 최신 이슈를 분석하여 보고서를 생성합니다.
      </Text>
    </View>
  );

  return (
    <View style={styles.container}>
      <Stack.Screen
        options={{
          headerShown: true,
          headerTitle: '',
          headerLeft: () => <HeaderLeftGoBack />,
          headerBackground: () => <View style={{ flex: 1, backgroundColor: '#f8f9fa' }} />,
          headerShadowVisible: false,
        }}
      />
      
      <FlatList
        data={reports}
        renderItem={({ item, index }) => (
          <Link 
            href={{
              pathname: `/report/[id]` as any,
              params: { 
                id: item.id,
                title: item.title,
                tags: JSON.stringify(item.tags),
                content: item.content || '',
                aiReport: item.aiReport || '',
                sources: JSON.stringify(item.sources || [])
              }
            }} 
            asChild
          >
            <ReportItem
              id={item.id}
              title={item.title}
              tags={item.tags}
            />
          </Link>
        )}
        keyExtractor={(item, index) => `${item.id}-${index}`}
        contentContainerStyle={styles.listContainer}
        showsVerticalScrollIndicator={false}
        ListFooterComponent={renderFooter}
        ListEmptyComponent={renderEmptyComponent}
        // onEndReached와 onEndReachedThreshold는 현재 사용하지 않음
        // onEndReached={loadMoreReports}
        // onEndReachedThreshold={0.1}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
  },
  listContainer: {
    paddingBottom: 16,
  },
  headerContainer: {
    paddingHorizontal: 20,
    paddingVertical: 20,
    backgroundColor: '#fff',
    marginBottom: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#e9ecef',
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 5,
  },
  headerSubtitle: {
    fontSize: 16,
    color: '#666',
  },
  footerLoader: {
    padding: 20,
    alignItems: 'center',
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 50,
  },
  emptyTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 10,
  },
  emptyText: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
  },
});