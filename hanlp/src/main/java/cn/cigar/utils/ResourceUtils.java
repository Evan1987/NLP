package cn.cigar.utils;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

/**
 * @author zhaochengming
 * @date 2021/1/18 10:29
 */
public class ResourceUtils {

    private static final String HANLP_PROPERTIES = "hanlp.properties";

    private InputStream readFileFromResource(String fileName) throws FileNotFoundException {
        InputStream stream = getClass().getClassLoader().getResourceAsStream(fileName);
        if(stream == null)
            throw new FileNotFoundException("file not found for " + fileName);
        return stream;
    }

    public static Properties loadProperties() throws IOException {
        ResourceUtils resourceUtils = new ResourceUtils();
        InputStream inputStream = resourceUtils.readFileFromResource(HANLP_PROPERTIES);
        Properties properties = new Properties();
        properties.load(inputStream);
        return properties;
    }
}
