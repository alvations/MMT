package eu.modernmt;

import eu.modernmt.aligner.Aligner;
import eu.modernmt.aligner.SymmetrizedAligner;
import eu.modernmt.aligner.fastalign.FastAlign;
import eu.modernmt.config.EngineConfig;
import eu.modernmt.constants.Const;
import eu.modernmt.context.ContextAnalyzer;
import eu.modernmt.context.lucene.LuceneAnalyzer;
import eu.modernmt.decoder.Decoder;
import eu.modernmt.decoder.moses.MosesDecoder;
import eu.modernmt.decoder.moses.MosesINI;
import eu.modernmt.io.Paths;
import eu.modernmt.processing.Postprocessor;
import eu.modernmt.processing.Preprocessor;
import eu.modernmt.vocabulary.Vocabulary;
import eu.modernmt.vocabulary.rocksdb.RocksDBVocabulary;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;

import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.util.Locale;
import java.util.Map;

/**
 * Created by davide on 19/04/16.
 */
public class Engine implements Closeable {

    public static final String ENGINE_CONFIG_PATH = "engine.ini";
    private static final String VOCABULARY_MODEL_PATH = Paths.join("models", "vocabulary");

    public static File getRootPath(String engine) {
        return new File(Const.fs.engines, engine);
    }

    public static File getConfigFile(String engine) {
        File root = new File(Const.fs.engines, engine);
        return new File(root, ENGINE_CONFIG_PATH);
    }

    private final EngineConfig config;
    private final int threads;
    private final File root;
    private final File runtime;
    private final String name;

    private Decoder decoder = null;
    private Aligner aligner = null;
    private Preprocessor preprocessor = null;
    private Postprocessor postprocessor = null;
    private ContextAnalyzer contextAnalyzer = null;
    private Vocabulary vocabulary = null;

    public Engine(EngineConfig config, int threads) {
        this.config = config;
        this.threads = threads;
        this.name = config.getName();
        this.root = new File(Const.fs.engines, config.getName());
        this.runtime = new File(Const.fs.runtime, name);
    }

    public EngineConfig getConfig() {
        return config;
    }

    public String getName() {
        return name;
    }

    public Decoder getDecoder() {
        if (decoder == null) {
            synchronized (this) {
                if (decoder == null) {
                    try {
                        File iniTemplate = Paths.join(root, "models", "moses.ini");
                        MosesINI mosesINI = MosesINI.load(iniTemplate, root);

                        Map<String, float[]> featureWeights = config.getDecoderConfig().getWeights();
                        if (featureWeights != null)
                            mosesINI.setWeights(featureWeights);

                        mosesINI.setThreads(threads);

                        File inifile = new File(runtime, "moses.ini");
                        FileUtils.write(inifile, mosesINI.toString(), false);
                        decoder = new MosesDecoder(inifile);
                    } catch (IOException e) {
                        throw new LazyLoadException(e);
                    }
                }
            }
        }

        return decoder;
    }

    public Aligner getAligner() {
        if (aligner == null) {
            synchronized (this) {
                if (aligner == null) {
                    try {
                        File modelDirectory = Paths.join(root, "models", "phrase_tables");
                        File fwdModelFile = new File(modelDirectory, "model.align.fwd");
                        File bwdModelFile = new File(modelDirectory, "model.align.bwd");

                        FastAlign fwdModel = new FastAlign(fwdModelFile);
                        FastAlign bwdModel = new FastAlign(bwdModelFile);

                        aligner = new SymmetrizedAligner(fwdModel, bwdModel);
                    } catch (IOException e) {
                        throw new LazyLoadException(e);
                    }
                }
            }
        }

        return aligner;
    }

    public Preprocessor getPreprocessor() {
        if (preprocessor == null) {
            synchronized (this) {
                if (preprocessor == null) {
                    try {
                        preprocessor = new Preprocessor(getSourceLanguage(), getTargetLanguage(), getVocabulary());
                    } catch (IOException e) {
                        throw new LazyLoadException(e);
                    }
                }
            }
        }

        return preprocessor;
    }

    public Postprocessor getPostprocessor() {
        if (postprocessor == null) {
            synchronized (this) {
                if (postprocessor == null) {
                    try {
                        postprocessor = new Postprocessor(getSourceLanguage(), getTargetLanguage(), getVocabulary());
                    } catch (IOException e) {
                        throw new LazyLoadException(e);
                    }
                }
            }
        }

        return postprocessor;
    }

    public ContextAnalyzer getContextAnalyzer() {
        if (contextAnalyzer == null) {
            synchronized (this) {
                if (contextAnalyzer == null) {
                    try {
                        File indexPath = Paths.join(root, "models", "context", "index");
                        this.contextAnalyzer = new LuceneAnalyzer(indexPath);
                    } catch (IOException e) {
                        throw new LazyLoadException(e);
                    }
                }
            }
        }

        return contextAnalyzer;
    }

    public Vocabulary getVocabulary() {
        if (vocabulary == null) {
            synchronized (this) {
                if (vocabulary == null) {
                    try {
                        File model = new File(this.root, VOCABULARY_MODEL_PATH);
                        vocabulary = new RocksDBVocabulary(model);
                    } catch (IOException e) {
                        throw new LazyLoadException(e);
                    }
                }
            }
        }

        return vocabulary;
    }

    public Locale getSourceLanguage() {
        return config.getSourceLanguage();
    }

    public Locale getTargetLanguage() {
        return config.getTargetLanguage();
    }

    public File getRootPath() {
        return root;
    }

    public File getRuntimeFolder(String folderName, boolean ensure) throws IOException {
        File folder = new File(this.runtime, folderName);

        if (ensure) {
            FileUtils.deleteDirectory(folder);
            FileUtils.forceMkdir(folder);
        }

        return folder;
    }

    @Override
    public void close() {
        IOUtils.closeQuietly(preprocessor);
        IOUtils.closeQuietly(postprocessor);

        IOUtils.closeQuietly(decoder);
        IOUtils.closeQuietly(aligner);
        IOUtils.closeQuietly(contextAnalyzer);
        IOUtils.closeQuietly(vocabulary);
    }

}
