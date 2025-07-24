import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons'; // 아이콘 사용을 위한 라이브러리

// ReportItem 컴포넌트가 받을 props의 타입을 정의합니다.
interface ReportItemProps {
  title: string;
  tags: string[];
}

/**
 * 보고서 목록의 개별 항목을 표시하는 컴포넌트입니다.
 * @param title - 보고서 제목
 * @param tags - 보고서 관련 태그 배열
 */
const ReportItem: React.FC<ReportItemProps> = ({ title, tags }) => {
  return (
    <View style={styles.container}>
      <Ionicons name="document-text-outline" size={24} color="#333" style={styles.icon} />
      <View style={styles.contentContainer}>
        <Text style={styles.title} numberOfLines={2}>{title}</Text>
        <View style={styles.tagContainer}>
          {tags.map((tag, index) => (
            <View key={index} style={styles.tag}>
              <Text style={styles.tagText}>#{tag}</Text>
            </View>
          ))}
        </View>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    marginHorizontal: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 3,
  },
  icon: {
    marginRight: 12,
  },
  contentContainer: {
    flex: 1,
  },
  title: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 8,
    color: '#333',
  },
  tagContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  tag: {
    backgroundColor: '#FBCEB1',
    borderRadius: 8,
    paddingVertical: 4,
    paddingHorizontal: 8,
    marginRight: 6,
    marginBottom: 6,
  },
  tagText: {
    color: '#000',
    fontSize: 12,
    marginBottom: 2,
  },
});

export default ReportItem;
