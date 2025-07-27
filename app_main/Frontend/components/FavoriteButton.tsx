// components/FavoriteButton.tsx
import React from 'react';
import { Pressable, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

interface FavoriteButtonProps {
  isFavorite: boolean;
  onPress: () => void;
}

export default function FavoriteButton({ isFavorite, onPress }: FavoriteButtonProps) {
  return (
    <Pressable onPress={onPress} style={styles.button}>
      <Ionicons
        name={isFavorite ? 'heart' : 'heart-outline'} // 꽉 찬 하트 vs 빈 하트
        size={28}
        color={isFavorite ? '#FF0000' : '#888'} // 즐겨찾기 시 빨간색
      />
    </Pressable>
  );
}

const styles = StyleSheet.create({
  button: {
    paddingHorizontal: 15, // 버튼 영역을 넓혀 터치하기 쉽게
    justifyContent: 'center',
    alignItems: 'center',
    height: '100%', // 헤더 높이에 맞춤
  },
});