package cn.cigar.chap01;


import com.hankcs.hanlp.HanLP;

/**
 * @author zhaochengming
 * @date 2021/1/17 22:36
 */
public class HelloWorld {
    public static void main(String[] args) {
        HanLP.Config.enableDebug();
        System.out.println(HanLP.segment("王国维和服务员"));
    }
}
