// utils/favorites.ts
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Alert } from 'react-native';

// 즐겨찾기 항목의 인터페이스를 정의합니다.
// ReportScreen의 SearchResponse와 유사하지만,
// 즐겨찾기 목록에서 필요한 최소한의 정보를 가집니다.
export interface FavoriteReport {
  id: string; // 고유 ID (제목 등을 해싱하거나, 백엔드에서 제공하는 ID 사용)
  title: string;
  tags: string[];
  // 필요하다면 원본 보고서 내용을 담을 수도 있지만,
  // 여기서는 최소한의 정보만 저장하고, 클릭 시 다시 API 호출하여 상세 정보를 가져오는 것을 권장합니다.
  // 여기서는 간단하게 searchData (JSON 문자열)를 저장하여 재활용합니다.
  searchData: string; // ReportScreen으로 넘겨줄 원본 JSON 문자열
}

const FAVORITES_KEY = 'my_favorite_reports';

// 모든 즐겨찾기 보고서 불러오기
export const getFavoriteReports = async (): Promise<FavoriteReport[]> => {
  try {
    const jsonValue = await AsyncStorage.getItem(FAVORITES_KEY);
    return jsonValue != null ? JSON.parse(jsonValue) : [];
  } catch (e) {
    console.error("Failed to load favorites.", e);
    Alert.alert("오류", "즐겨찾기 목록을 불러오는 데 실패했습니다.");
    return [];
  }
};

// 즐겨찾기 보고서 저장 (덮어쓰기)
const saveFavoriteReports = async (reports: FavoriteReport[]) => {
  try {
    const jsonValue = JSON.stringify(reports);
    await AsyncStorage.setItem(FAVORITES_KEY, jsonValue);
  } catch (e) {
    console.error("Failed to save favorites.", e);
    Alert.alert("오류", "즐겨찾기 목록을 저장하는 데 실패했습니다.");
  }
};

// 보고서를 즐겨찾기에 추가 또는 제거
export const toggleFavorite = async (report: FavoriteReport): Promise<boolean> => {
  const currentFavorites = await getFavoriteReports();
  const isCurrentlyFavorite = currentFavorites.some(fav => fav.id === report.id);

  if (isCurrentlyFavorite) {
    // 이미 즐겨찾기에 있으면 제거
    const newFavorites = currentFavorites.filter(fav => fav.id !== report.id);
    await saveFavoriteReports(newFavorites);
    Alert.alert("알림", `${report.title} 보고서가 즐겨찾기에서 제거되었습니다.`);
    return false;
  } else {
    // 즐겨찾기에 없으면 추가 (최대 20개 제한)
    if (currentFavorites.length >= 20) {
      Alert.alert("알림", "즐겨찾기는 최대 20개까지만 저장할 수 있습니다.");
      return false;
    }
    const newFavorites = [...currentFavorites, report];
    await saveFavoriteReports(newFavorites);
    Alert.alert("알림", `${report.title} 보고서가 즐겨찾기에 추가되었습니다.`);
    return true;
  }
};

// 특정 보고서가 즐겨찾기에 있는지 확인
export const isReportFavorite = async (reportId: string): Promise<boolean> => {
  const currentFavorites = await getFavoriteReports();
  return currentFavorites.some(fav => fav.id === reportId);
};