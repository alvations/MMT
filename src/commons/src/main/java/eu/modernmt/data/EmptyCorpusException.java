package eu.modernmt.data;

/**
 * Created by davide on 06/09/16.
 */
public class EmptyCorpusException extends DataManagerException {

    public EmptyCorpusException() {
        super("Failed to import empty or poor quality corpus");
    }

}
