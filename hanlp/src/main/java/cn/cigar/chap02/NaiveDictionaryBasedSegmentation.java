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

        String text = "江西鄱阳湖干枯，中国最大淡水湖变成大草原";
        System.out.println("正向最大：");
        System.out.println(longestForwardSegment(text, dictionary));

        System.out.println("逆向最大：");
        System.out.println(longestBackwardSegment(text, dictionary));
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

    // 逆向最长匹配
    public static List<String> longestBackwardSegment(String text, Map<String, CoreDictionary.Attribute> dictionary) {
        List<String> words = new LinkedList<>();
        int i = text.length() - 1;
        while(i >= 0) {
            String longestWord = text.substring(i, i + 1);
            for(int j = i; j >= 0; j --) {
                String word = text.substring(j, i + 1);
                if(dictionary.containsKey(word))
                    longestWord = word;
            }
            words.add(0, longestWord);
            i -= longestWord.length();
        }
        return words;
    }

    // 统计分词结果中单字数量
    public static int countSingleChar(List<String> tokens) {
        int num = 0;
        for(String w: tokens)
            if(w.length() == 1)
                num ++;
        return num;
    }

    // 双向最长匹配
    public static List<String> longestBidirectionalSegment(String text, Map<String, CoreDictionary.Attribute> dictionary) {
        List<String> forwardRes = longestForwardSegment(text, dictionary);
        List<String> backwardRes = longestBackwardSegment(text, dictionary);
        if(forwardRes.size() == backwardRes.size()) {
            return countSingleChar(forwardRes) < countSingleChar(backwardRes) ? forwardRes : backwardRes;
        }

        return forwardRes.size() < backwardRes.size() ? forwardRes : backwardRes;
    }
}
