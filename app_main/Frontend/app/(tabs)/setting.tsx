import { View, Text, StyleSheet } from 'react-native';
import { Stack } from 'expo-router';
import HeaderLeftGoBack from '../../components/HeaderLeftGoBack'; // HeaderLeftGoBack.tsx 파일의 상대 경로

export default function SettingScreen() {
  return (
    <View style={styles.container}>
      {/* Stack.Screen 컴포넌트를 사용하여 이 화면의 헤더를 개별적으로 설정합니다.
        이 설정은 이 파일(like.tsx)에만 적용됩니다.
      */}
      <Stack.Screen
        options={{
          headerShown: true, // 헤더를 보이도록 설정합니다.
          headerTitle: '', // 헤더 중앙에 표시될 제목입니다.
          headerLeft: () => <HeaderLeftGoBack />, // 헤더 왼쪽에 표시될 컴포넌트를 지정합니다.
          headerTitleAlign: 'center', // 제목을 중앙에 정렬합니다.
          headerShadowVisible: false, // 헤더 아래의 그림자를 제거합니다.
        }}
      />

      <Text>좋아요 페이지 콘텐츠</Text>
      {/* 여기에 좋아요 페이지의 나머지 콘텐츠를 추가하세요. */}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#fff',
  },
});
