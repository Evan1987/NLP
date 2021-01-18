package cn.cigar.chap02;

import cn.cigar.utils.ResourceUtils;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.dictionary.CoreDictionary;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.*;

/**
 * @author zhaochengming
 * @date 2021/1/17 22:41
 */
public class NaiveDictionaryBasedSegmentation {

    public static void main(String[] args) throws IOException {

        String hanlpRoot = ResourceUtils.loadProperties().getProperty("root");

        TreeMap<String, CoreDictionary.Attribute> dictionary =
                IOUtil.loadDictionary(Paths.get(hanlpRoot, "/data/dictionary/CoreNatureDictionary.mini.txt").toString());

        System.out.println(longestForwardSegment("江西鄱阳湖干枯，中国最大淡水湖变成大草原", dictionary));
    }

    // 完全切分
    public static List<String> fullySegment(String text, Map<String, CoreDictionary.Attribute> dictionary) {
        List<String> words = new LinkedList<>();
        for(int i = 0; i < text.length(); i ++) {
            for(int j = i + 1; j <= text.length(); j ++) {
                String word = text.substring(i, j);
                if(dictionary.containsKey(word))
                    words.add(word);
            }
        }
        return words;
    }

    // 正向最长匹配
    public static List<String> longestForwardSegment(String text, Map<String, CoreDictionary.Attribute> dictionary) {
        List<String> words = new LinkedList<>();
        int i = 0;
        while(i < text.length()) {
            String longestWord = text.substring(i, i + 1);
            for(int j = i + 1; j <= text.length(); j ++) {
                String word = text.substring(i, j);
                if(dictionary.containsKey(word))
                    longestWord = word;
            }
            words.add(longestWord);
            i += longestWord.length();
        }
        return words;
    }
}
