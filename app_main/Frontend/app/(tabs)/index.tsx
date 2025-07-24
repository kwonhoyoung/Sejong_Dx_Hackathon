import { Image } from 'expo-image';
import { Platform, StyleSheet } from 'react-native';

import { View, Text, TextInput } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useState } from 'react';


export default function HomeScreen() {

    const [search, setSearch] = useState('');

    return (
        <View style={styles.container}>
            <View style={styles.searchContainer}>
                <Ionicons name='search' size={24} color='black' style={styles.searchIcon}/>
                <TextInput 
                    placeholder='Search 검색' 
                    style={styles.input}
                    value={search}
                    onChangeText={setSearch} />
            </View>
        </View>
);
}

const styles = StyleSheet.create({
container: {
  flex: 1,       
  paddingTop: 300,
  alignItems: 'center',
  backgroundColor: '#fff',
},
searchContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    width: '80%',
    height: 50,
    borderWidth: 0,
    paddingHorizontal: 15,
    borderRadius: 50,
    backgroundColor: '#FBCEB1',
    fontSize: 18,
},
searchIcon: {
    position: 'absolute',
    marginLeft: 15,
},
input: {
    flex: 1,
    fontSize: 20,
    marginLeft: 30,
},
});