import { useRouter } from 'expo-router';
import { Pressable, StyleSheet, Text, View } from 'react-native'; // View 추가
import AntDesign from '@expo/vector-icons/AntDesign';
import React from 'react'; // React import 추가

/**
 * 헤더 좌측에 위치하여 뒤로가기 기능을 수행하는 버튼 컴포넌트입니다.
 * Expo Router의 useRouter 훅을 사용하여 이전 페이지로 이동합니다.
 */
// title prop을 받도록 수정
export default function HeaderLeftGoBack({ title }: { title?: string }) {
  const router = useRouter();

  const handlePress = () => {
    if (router.canGoBack()) {
      router.back();
    } else {
      router.push('./index');
    }
  };

  return (
    <Pressable onPress={handlePress} style={styles.button}>
      <AntDesign name="left" size={24} color="#000" />
      <Text style={styles.text}>홈</Text>
      {/* title prop이 있다면 띄웁니다. */}
      {title && (
        <>
          <Text style={styles.separatorText}>|</Text>
          <Text numberOfLines={1} style={styles.titleText}>
            {title}
          </Text>
        </>
      )}
    </Pressable>
  );
}

const styles = StyleSheet.create({
  button: {
    flexDirection: 'row',
    paddingHorizontal: 16,
    justifyContent: 'center',
    height: '100%',
    alignItems: 'center',
  },
  text: {
    fontSize: 20,
    fontWeight: 'bold',
    marginLeft: 5,
    marginBottom: 3,
  },
  separatorText: {
    fontSize: 20,
    color: '#ccc',
    marginHorizontal: 8,
    marginBottom: 3,
  },
  titleText: {
    fontSize: 16, // "홈" 텍스트보다 약간 작게 설정하여 구별
    color: '#555',
    flexShrink: 1, // 텍스트가 길어질 때 줄어들도록 함
    marginBottom: 3,
  },
});