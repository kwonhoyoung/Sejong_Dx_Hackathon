import React, { useState, useEffect } from 'react';
import { View, StyleSheet, FlatList, Pressable, Text } from 'react-native';
import { Stack } from 'expo-router';
import ReportItem from '../../components/ReportItem'; // 방금 만든 ReportItem 컴포넌트
import HeaderLeftGoBack from '../../components/HeaderLeftGoBack'; // 이전 단계에서 만든 뒤로가기 컴포넌트


interface ReportData {
  id: string;
  title: string;
  tags: string[];
}

// 백엔드 API가 완성되기 전 사용할 임시 데이터
const MOCK_DATA: ReportData[] = [
  { id: '1', title: 'AI 리버싱 기술, 코드게이트 2025 핵심 화두 부상', tags: ['리버싱', 'AI리버싱', '코드게이트2025'] },
  { id: '2', title: '클라우드 보안의 미래, 서버리스 아키텍처의 역할', tags: ['클라우드', '보안', '서버리스'] },
  { id: '3', title: 'LLM을 활용한 효과적인 챗봇 개발 전략', tags: ['LLM', '챗봇', '자연어처리'] },
  { id: '4', title: '데이터 프라이버시, GDPR 준수를 위한 기술적 접근', tags: ['데이터', '프라이버시', 'GDPR'] },
  { id: '5', title: '양자 컴퓨팅이 암호화 기술에 미치는 영향 분석', tags: ['양자컴퓨팅', '암호화', '보안'] },
];

export default function LikeScreen() {
  const [reports, setReports] = useState<ReportData[]>([]);

  // 컴포넌트가 마운트될 때 데이터를 불러옵니다.
  // TODO: 추후 실제 API 호출 코드로 교체해야 합니다.
  useEffect(() => {
    // 현재는 임시 데이터를 사용합니다.
    setReports(MOCK_DATA);
    // 예시: fetch('https://your-api.com/reports')
    //   .then(res => res.json())
    //   .then(data => setReports(data));
  }, []);

  const handleEdit = () => {
    // '편집' 버튼을 눌렀을 때의 동작을 여기에 구현합니다.
    console.log('편집 버튼 클릭');
  };

  return (
    <View style={styles.container}>
      <Stack.Screen
        options={{
          headerShown: true,
          headerTitle: '', // 제목을 비워서 깔끔하게 만듭니다.
          headerLeft: () => <HeaderLeftGoBack />,
          headerRight: () => (
            <Pressable onPress={handleEdit} style={styles.editButton}>
              <Text style={styles.editButtonText}>편집</Text>
            </Pressable>
          ),
          headerBackground: () => <View style={{ flex: 1, backgroundColor: '#f8f9fa' }} />,
          headerShadowVisible: false,
        }}
      />
      <FlatList
        data={reports}
        renderItem={({ item }) => <ReportItem title={item.title} tags={item.tags} />}
        keyExtractor={item => item.id}
        contentContainerStyle={styles.listContainer}
        showsVerticalScrollIndicator={false}
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
    paddingTop: 16,
    paddingBottom: 16,
  },
  editButton: {
    marginRight: 16,
    padding: 8,
  },
  editButtonText: {
    fontSize: 20,
    color: '#000',
  },
});
