import { Tabs } from 'expo-router';
import React from 'react';
import { Platform } from 'react-native';

import { HapticTab } from '@/components/HapticTab';
import { IconSymbol } from '@/components/ui/IconSymbol';
import TabBarBackground from '@/components/ui/TabBarBackground';
import { Colors } from '@/constants/Colors';
import { useColorScheme } from '@/hooks/useColorScheme';
import FontAwesome from '@expo/vector-icons/FontAwesome';
import Feather from '@expo/vector-icons/Feather';
import AntDesign from '@expo/vector-icons/AntDesign';


export default function TabLayout() {
  const colorScheme = useColorScheme();

  return (
    <Tabs
      screenOptions={{
        tabBarActiveTintColor: Colors[colorScheme ?? 'light'].tint,
        headerShown: false,
        tabBarButton: HapticTab,
        tabBarBackground: TabBarBackground,
        tabBarStyle: Platform.select({
          ios: {
            // Use a transparent background on iOS to show the blur effect
            position: 'absolute',
          },
          default: {},
        }),
      }}>
      <Tabs.Screen
        name="index"
        options={{
          title: '홈',
          tabBarIcon: ({ color }) => <Feather name="home" size={28} color={color} />,
        }}
      />
        <Tabs.Screen
          name="report"
          options={{
            title: '보고서',
            //headerShown: true,
            tabBarIcon: ({ color }) => <AntDesign name="filetext1" size={24} color={color} />,
          }}
        />
        <Tabs.Screen
          name="like"
          options={{
            title: '즐겨찾기',
            //headerShown: true,
            tabBarIcon: ({ color }) => <FontAwesome name="bookmark-o" size={24} color={color} />,
          }}
        />
      <Tabs.Screen
        name="setting"
        options={{
          title: '설정',
          //headerShown: true,
          tabBarIcon: ({ color }) => <Feather name="settings" size={24} color={color} />,
        }}
      />
    </Tabs>
  );
}
