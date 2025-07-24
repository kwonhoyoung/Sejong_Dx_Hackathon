import { useRouter } from 'expo-router';
import { Pressable, StyleSheet, Text } from 'react-native';
import AntDesign from '@expo/vector-icons/AntDesign';

/**
 * 헤더 좌측에 위치하여 뒤로가기 기능을 수행하는 버튼 컴포넌트입니다.
 * Expo Router의 useRouter 훅을 사용하여 이전 페이지로 이동합니다.
 */
export default function HeaderLeftGoBack() {
  const router = useRouter();

  // 버튼을 누르면 router.back() 함수를 호출하여 이전 스크린으로 돌아갑니다.
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
    </Pressable>
  );
}

const styles = StyleSheet.create({
  button: {
    // iOS에서는 기본적으로 왼쪽에 여백이 있으나, 일관성을 위해 추가합니다.
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
});
