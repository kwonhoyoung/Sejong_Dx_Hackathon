import React, { useState, useEffect } from 'react';
import { View, StyleSheet, FlatList, Pressable, Text } from 'react-native';
import { Stack } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import ReportItem from '../../components/ReportItem';
import HeaderLeftGoBack from '@/components/HeaderLeftGoBack';

interface ReportData {
  id: string;
  title: string;
  tags: string[];
}

const MOCK_DATA: ReportData[] = [
  { id: '1', title: 'AI 리버싱 기술, 코드게이트 2025 핵심 화두 부상', tags: ['리버싱', 'AI리버싱', '코드게이트2025'] },
  { id: '2', title: '클라우드 보안의 미래, 서버리스 아키텍처의 역할', tags: ['클라우드', '보안', '서버리스'] },
  { id: '3', title: 'LLM을 활용한 효과적인 챗봇 개발 전략', tags: ['LLM', '챗봇', '자연어처리'] },
  { id: '4', title: '데이터 프라이버시, GDPR 준수를 위한 기술적 접근', tags: ['데이터', '프라이버시', 'GDPR'] },
  { id: '5', title: '양자 컴퓨팅이 암호화 기술에 미치는 영향 분석', tags: ['양자컴퓨팅', '암호화', '보안'] },
];

export default function ReportScreen() {
  const [reports, setReports] = useState<ReportData[]>([]);
  const [isEditMode, setIsEditMode] = useState(false); // 편집 모드 상태
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set()); // 선택된 항목 ID 저장

  useEffect(() => {
    setReports(MOCK_DATA);
  }, []);

  // 편집 모드 토글 함수
  const toggleEditMode = () => {
    if (isEditMode) {
      // 편집 모드를 나갈 때 선택 목록 초기화
      setSelectedIds(new Set());
    }
    setIsEditMode(!isEditMode);
  };

  // 항목 선택/해제 처리 함수
  const handleSelectReport = (id: string) => {
    const newSelectedIds = new Set(selectedIds);
    if (newSelectedIds.has(id)) {
      newSelectedIds.delete(id);
    } else {
      newSelectedIds.add(id);
    }
    setSelectedIds(newSelectedIds);
  };

  // 선택된 항목 삭제 함수
  const handleDeleteSelected = () => {
    if (selectedIds.size === 0) {
      // 선택된 항목이 없으면 아무것도 안 함
      toggleEditMode(); // 편집 모드 종료
      return;
    }
    const newReports = reports.filter(report => !selectedIds.has(report.id));
    setReports(newReports);
    toggleEditMode(); // 삭제 후 편집 모드 종료
  };

  return (
    <View style={styles.container}>
      <Stack.Screen
        options={{
          headerShown: true,
          headerTitle: '',
          headerShadowVisible: false,
          headerBackground: () => <View style={{ flex: 1, backgroundColor: '#f8f9fa' }} />,
          // 편집 모드에 따라 헤더 버튼을 동적으로 변경
          headerLeft: () => (
            isEditMode ? (
              <Pressable onPress={toggleEditMode} style={styles.headerButton}>
                <Text style={styles.headerButtonText}>취소</Text>
              </Pressable>
            ) : (
              <HeaderLeftGoBack />
            )
          ),
          headerRight: () => (
            isEditMode ? (
              <Pressable onPress={handleDeleteSelected} style={styles.headerButton}>
                <Ionicons name="trash-outline" size={28} color="#FF3B30" />
              </Pressable>
            ) : (
              <Pressable onPress={toggleEditMode} style={styles.headerButton}>
                <Text style={styles.headerButtonText}>편집</Text>
              </Pressable>
            )
          ),
        }}
      />
      <FlatList
        data={reports}
        renderItem={({ item }) => (
          <ReportItem
            id={item.id}
            title={item.title}
            tags={item.tags}
            isEditMode={isEditMode}
            isSelected={selectedIds.has(item.id)}
            onSelect={handleSelectReport}
          />
        )}
        keyExtractor={item => item.id}
        contentContainerStyle={styles.listContainer}
        showsVerticalScrollIndicator={false}
        // FlatList의 추가적인 최적화 속성
        extraData={{ isEditMode, selectedIds }}
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
  headerButton: {
    paddingHorizontal: 20,
    justifyContent: 'center',
    height: '100%',
  },
  headerButtonText: {
    fontSize: 20,
    color: '#000',
  },
});
