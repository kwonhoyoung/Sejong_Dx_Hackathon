import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Pressable,
  Linking,
  Alert
} from 'react-native';
import { Stack, useLocalSearchParams } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import HeaderLeftGoBack from '../../components/HeaderLeftGoBack'; // 경로 확인

export default function ReportDetailScreen() {
  const params = useLocalSearchParams();

  const {
    id,
    title, // title 값을 HeaderLeftGoBack으로 전달할 것입니다.
    tags: tagsParam,
    content,
    aiReport,
    sources: sourcesParam
  } = params;

  const tags = tagsParam ? JSON.parse(tagsParam as string) : [];
  const sources = sourcesParam ? JSON.parse(sourcesParam as string) : [];

  const openLink = async (url: string) => {
    try {
      const supported = await Linking.canOpenURL(url);
      if (supported) {
        await Linking.openURL(url);
      } else {
        Alert.alert('오류', '링크를 열 수 없습니다.');
      }
    } catch (error) {
      console.error('링크 열기 오류:', error);
      Alert.alert('오류', '링크를 여는 중 오류가 발생했습니다.');
    }
  };

  const renderTag = (tag: string, index: number) => (
    <View key={index} style={styles.tag}>
      <Text style={styles.tagText}>#{tag}</Text>
    </View>
  );

  const renderSource = (source: string, index: number) => (
    <Pressable
      key={index}
      style={styles.sourceItem}
      onPress={() => openLink(source)}
    >
      <Ionicons name="link-outline" size={16} color="#007AFF" />
      <Text style={styles.sourceText} numberOfLines={1}>
        {source}
      </Text>
      <Ionicons name="open-outline" size={16} color="#007AFF" />
    </Pressable>
  );

  return (
    <View style={styles.container}>
      <Stack.Screen
        options={{
          headerShown: true,
          headerTitle: () => (
            <Text numberOfLines={1} style={styles.headerTitleText}>
              {title}
            </Text>
          ),
          // 여기에 title prop을 전달합니다.
          headerLeft: () => <HeaderLeftGoBack title={title as string} />,
          headerBackground: () => <View style={{ flex: 1, backgroundColor: '#fff' }} />,
          headerShadowVisible: false,
        }}
      />

      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {/* 헤더 섹션 (여기서는 title을 제거해도 됩니다, 아니면 중복해서 표시됨) */}
        <View style={styles.header}>
          {/* <Text style={styles.title}>{title}</Text> -- 이 부분은 이제 헤더 타이틀로 이동했으므로 필요없을 수 있습니다. */}

          {/* 태그 섹션 */}
          <View style={styles.tagsContainer}>
            {tags.map((tag: string, index: number) => renderTag(tag, index))}
          </View>
        </View>

        {/* 정리된 내용 섹션 */}
        {content && (
          <View style={styles.section}>
            <View style={styles.sectionHeader}>
              <Ionicons name="document-text-outline" size={20} color="#333" />
              <Text style={styles.sectionTitle}>📋 정리된 내용</Text>
            </View>
            <Text style={styles.contentText}>{content}</Text>
          </View>
        )}

        {/* AI 분석 리포트 섹션 */}
        {aiReport && (
          <View style={styles.section}>
            <View style={styles.sectionHeader}>
              <Ionicons name="analytics-outline" size={20} color="#007AFF" />
              <Text style={styles.sectionTitle}>🤖 AI 분석 리포트</Text>
            </View>
            <View style={styles.aiReportContainer}>
              <Text style={styles.aiReportText}>{aiReport}</Text>
            </View>
          </View>
        )}

        {/* 출처 링크 섹션 */}
        {sources.length > 0 && (
          <View style={styles.section}>
            <View style={styles.sectionHeader}>
              <Ionicons name="library-outline" size={20} color="#28a745" />
              <Text style={styles.sectionTitle}>🔗 출처 링크</Text>
            </View>
            <View style={styles.sourcesContainer}>
              {sources.map((source: string, index: number) => renderSource(source, index))}
            </View>
          </View>
        )}

        {/* 추가 정보 섹션 */}
        <View style={styles.infoSection}>
          <Text style={styles.infoText}>
            💡 이 보고서는 AI가 최신 정보를 분석하여 생성한 내용입니다.
          </Text>
          <Text style={styles.infoText}>
            📅 생성 시간: {new Date().toLocaleString('ko-KR')}
          </Text>
        </View>
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
  },
  scrollContent: {
    paddingBottom: 30,
  },
  header: {
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#e9ecef',
    backgroundColor: '#f8f9fa',
  },
  headerTitleText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    textAlign: 'center',
    flexShrink: 1,
  },
  tagsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  tag: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 15,
  },
  tagText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '600',
  },
  section: {
    margin: 20,
    marginBottom: 0,
  },
  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 15,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginLeft: 8,
  },
  contentText: {
    fontSize: 16,
    lineHeight: 24,
    color: '#444',
    backgroundColor: '#f8f9fa',
    padding: 15,
    borderRadius: 10,
    borderLeftWidth: 4,
    borderLeftColor: '#007AFF',
  },
  aiReportContainer: {
    backgroundColor: '#e3f2fd',
    padding: 15,
    borderRadius: 10,
    borderLeftWidth: 4,
    borderLeftColor: '#007AFF',
  },
  aiReportText: {
    fontSize: 15,
    lineHeight: 22,
    color: '#333',
  },
  sourcesContainer: {
    gap: 10,
  },
  sourceItem: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f8f9fa',
    padding: 12,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#e9ecef',
    gap: 10,
  },
  sourceText: {
    flex: 1,
    fontSize: 14,
    color: '#007AFF',
    textDecorationLine: 'underline',
  },
  infoSection: {
    margin: 20,
    padding: 15,
    backgroundColor: '#fff3cd',
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#ffeaa7',
  },
  infoText: {
    fontSize: 13,
    color: '#856404',
    marginBottom: 5,
    lineHeight: 18,
  },
});