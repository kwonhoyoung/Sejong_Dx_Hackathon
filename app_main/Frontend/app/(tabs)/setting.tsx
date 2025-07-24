import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Switch,
  TouchableOpacity,
  Alert,
  ScrollView,
  Platform,
  TextInput,
} from 'react-native';
import { Stack } from 'expo-router';
import HeaderLeftGoBack from '../../components/HeaderLeftGoBack';

let notificationStorage = {
  notificationTimes: [
    { id: '1', time: '09:41', enabled: true },
    { id: '2', time: '16:42', enabled: true },
  ],
  allNotificationsPaused: false,
};

interface NotificationTime {
  id: string;
  time: string;
  enabled: boolean;
}

export default function SettingScreen() {
  const [notificationTimes, setNotificationTimes] = useState<NotificationTime[]>(
    notificationStorage.notificationTimes
  );
  const [allNotificationsPaused, setAllNotificationsPaused] = useState(
    notificationStorage.allNotificationsPaused
  );
  const [newTime, setNewTime] = useState('');

  const toggleNotificationTime = (id: string) => {
    const updatedTimes = notificationTimes.map(item =>
      item.id === id ? { ...item, enabled: !item.enabled } : item
    );
    setNotificationTimes(updatedTimes);
    notificationStorage.notificationTimes = updatedTimes;

    console.log('알림 시간 설정 변경:', notificationStorage.notificationTimes);
  };

  const toggleAllNotifications = () => {
    const newPausedState = !allNotificationsPaused;
    setAllNotificationsPaused(newPausedState);
    notificationStorage.allNotificationsPaused = newPausedState;

    console.log('🔕 전체 알림 상태 변경:', newPausedState);

    Alert.alert(
      newPausedState ? '알림 중지' : '알림 재개',
      newPausedState ? '모든 알림이 일시중지되었습니다.' : '알림이 다시 활성화되었습니다.'
    );
  };

  const clearCache = () => {
    Alert.alert(
      '캐시 삭제',
      '모든 캐시 데이터가 삭제됩니다. 계속하시겠습니까?',
      [
        { text: '취소', style: 'cancel' },
        {
          text: '삭제',
          style: 'destructive',
          onPress: () => {
            console.log('🧹 캐시 삭제 실행');
            Alert.alert('완료', '캐시가 성공적으로 삭제되었습니다.');
          },
        },
      ]
    );
  };

  const addNotificationTime = () => {
    if (!newTime.match(/^\d{2}:\d{2}$/)) {
      Alert.alert('시간 형식 오류', '시간은 HH:MM 형식으로 입력해주세요.');
      return;
    }

    const newId = Date.now().toString();
    const newEntry = { id: newId, time: newTime, enabled: false };
    const updated = [...notificationTimes, newEntry];

    setNotificationTimes(updated);
    notificationStorage.notificationTimes = updated;
    console.log('➕ 알림 추가:', newTime);
    setNewTime('');
  };

  // 삭제 함수 추가
  const deleteNotificationTime = (id: string) => {
    const filtered = notificationTimes.filter(item => item.id !== id);
    setNotificationTimes(filtered);
    notificationStorage.notificationTimes = filtered;
    console.log('➖ 알림 삭제:', id);
  };

  return (
    <View style={styles.container}>
      <Stack.Screen
        options={{
          headerShown: true,
          headerTitle: '',
          headerLeft: () => <HeaderLeftGoBack />,
          headerTitleAlign: 'center',
          headerShadowVisible: false,
          headerBackground: () => <View style={{ flex: 1, backgroundColor: '#f8f9fa' }} />,
        }}
      />

      <ScrollView style={styles.scrollView} showsVerticalScrollIndicator={false}>
        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>모든 알람 일시 중지</Text>
            <Switch
              value={allNotificationsPaused}
              onValueChange={toggleAllNotifications}
              trackColor={{ false: '#e0e0e0', true: '#34C759' }}
              thumbColor={
                Platform.OS === 'ios' ? '#ffffff' : allNotificationsPaused ? '#ffffff' : '#f4f3f4'
              }
            />
          </View>
        </View>

        <View style={styles.section}>
          {/* 알림 시간 설정 헤더 + 입력 필드 */}
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>알림 시간 설정</Text>
            <View style={styles.addRow}>
              <TextInput
                placeholder="13:45"
                value={newTime}
                onChangeText={setNewTime}
                style={styles.input}
                keyboardType="numeric"
                maxLength={5}
              />
              <TouchableOpacity onPress={addNotificationTime} style={styles.addButton}>
                <Text style={styles.addButtonText}>추가</Text>
              </TouchableOpacity>
            </View>
          </View>

          {notificationTimes.map(timeItem => (
            <View key={timeItem.id} style={styles.timeItem}>
              <Text style={styles.timeText}>{timeItem.time}</Text>

              <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                <Switch
                  value={timeItem.enabled && !allNotificationsPaused}
                  onValueChange={() => toggleNotificationTime(timeItem.id)}
                  disabled={allNotificationsPaused}
                  trackColor={{ false: '#e0e0e0', true: '#34C759' }}
                  thumbColor={
                    Platform.OS === 'ios'
                      ? '#ffffff'
                      : timeItem.enabled
                      ? '#ffffff'
                      : '#f4f3f4'
                  }
                />
                <TouchableOpacity
                  onPress={() =>
                    Alert.alert(
                      '삭제 확인',
                      `${timeItem.time} 알림을 삭제하시겠습니까?`,
                      [
                        { text: '취소', style: 'cancel' },
                        {
                          text: '삭제',
                          style: 'destructive',
                          onPress: () => deleteNotificationTime(timeItem.id),
                        },
                      ]
                    )
                  }
                  style={styles.deleteButton}
                >
                  <Text style={styles.deleteButtonText}>🗑</Text>
                </TouchableOpacity>
              </View>
            </View>
          ))}
        </View>

        <View style={styles.section}>
          <TouchableOpacity style={styles.cacheButton} onPress={clearCache}>
            <Text style={styles.cacheButtonText}>캐시 삭제</Text>
            <View style={styles.trashIcon}>
              <Text style={styles.trashIconText}>🗑</Text>
            </View>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
  },
  scrollView: {
    flex: 1,
  },
  section: {
    backgroundColor: '#ffffff',
    marginHorizontal: 16,
    marginVertical: 8,
    borderRadius: 12,
    padding: 16,
    elevation: 1,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '500',
    color: '#333',
  },
  timeItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  timeText: {
    fontSize: 18,
    color: '#333',
    fontWeight: '500',
  },
  addRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  input: {
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 8,
    paddingHorizontal: 10,
    paddingVertical: 4,
    fontSize: 14,
    width: 60,
  },
  addButton: {
    backgroundColor: '#34C759',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
  },
  addButtonText: {
    color: '#fff',
    fontWeight: '600',
    fontSize: 14,
  },
  cacheButton: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 4,
  },
  cacheButtonText: {
    fontSize: 16,
    color: '#333',
    fontWeight: '500',
  },
  trashIcon: {
    width: 24,
    height: 24,
    justifyContent: 'center',
    alignItems: 'center',
  },
  trashIconText: {
    fontSize: 18,
  },

  deleteButton: {
    marginLeft: 12,
    padding: 6,
    justifyContent: 'center',
    alignItems: 'center',
  },
  deleteButtonText: {
    fontSize: 18,
    color: '#ff3b30', // iOS 빨간색 느낌
  },
});
