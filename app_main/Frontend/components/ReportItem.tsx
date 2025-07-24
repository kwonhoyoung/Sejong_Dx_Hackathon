import React from 'react';
import { View, Text, StyleSheet, Pressable } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

// ReportItem 컴포넌트가 받을 props의 타입을 수정합니다.
// 편집 관련 props 뒤에 '?'를 붙여 선택적으로 만듭니다.
interface ReportItemProps {
  id: string; // id는 여전히 필수입니다.
  title: string;
  tags: string[];
  isEditMode?: boolean; // 선택적 prop
  isSelected?: boolean; // 선택적 prop
  onSelect?: (id: string) => void; // 선택적 prop
}

/**
 * 보고서 목록의 개별 항목을 표시하는 컴포넌트입니다.
 * 편집 모드를 선택적으로 지원합니다.
 */
const ReportItem: React.FC<ReportItemProps> = ({
  id,
  title,
  tags,
  // props가 제공되지 않았을 때 사용할 기본값을 설정합니다.
  isEditMode = false,
  isSelected = false,
  onSelect = () => {}, // 기본값으로 빈 함수를 설정하여 오류 방지
}) => {
  return (
    // isEditMode가 true일 때만 항목 전체를 누를 수 있도록 합니다.
    <Pressable style={styles.outerContainer} onPress={() => isEditMode && onSelect(id)} disabled={!isEditMode}>
      <View style={styles.container}>
        {/* 편집 모드일 때만 선택 버튼을 표시합니다. */}
        {isEditMode && (
          <Pressable onPress={() => onSelect(id)} style={styles.checkboxContainer}>
            <Ionicons
              name={isSelected ? 'checkmark-circle' : 'ellipse-outline'}
              size={24}
              color={isSelected ? '#e9946d' : '#ccc'}
            />
          </Pressable>
        )}
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
    </Pressable>
  );
};

const styles = StyleSheet.create({
    outerContainer: {
        marginHorizontal: 16,
        marginBottom: 12,
    },
    container: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: '#fff',
        borderRadius: 12,
        padding: 16,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.05,
        shadowRadius: 4,
        elevation: 3,
    },
    checkboxContainer: {
        marginRight: 12,
        padding: 4,
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
        backgroundColor: '#f0f0f0',
        borderRadius: 8,
        paddingVertical: 4,
        paddingHorizontal: 8,
        marginRight: 6,
        marginBottom: 6,
    },
    tagText: {
        color: '#555',
        fontSize: 12,
    },
});

export default ReportItem;
