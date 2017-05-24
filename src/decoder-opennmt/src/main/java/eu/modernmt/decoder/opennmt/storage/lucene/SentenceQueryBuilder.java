package eu.modernmt.decoder.opennmt.storage.lucene;

import eu.modernmt.io.TokensOutputStream;
import eu.modernmt.model.Sentence;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;

import java.io.IOException;

/**
 * Created by davide on 24/05/17.
 */
class SentenceQueryBuilder {

    public Query build(Sentence sentence) {
        int length = sentence.getWords().length;
        boolean isLongQuery = length > 4;

        BooleanQuery termsQuery = new BooleanQuery();
        loadTerms(sentence, isLongQuery ? Analyzers.getLongQueryAnalyzer() : Analyzers.getShortQueryAnalyzer(), termsQuery);

        int minMatches = isLongQuery ? Math.max(1, (int) (length * .33)) : 1;
        termsQuery.setMinimumNumberShouldMatch(minMatches);

        return termsQuery;
    }

    private static void loadTerms(Sentence sentence, Analyzer analyzer, BooleanQuery output) {
        String text = TokensOutputStream.toString(sentence, false, true);

        try {
            TokenStream stream = analyzer.tokenStream("content", text);
            CharTermAttribute charTermAttribute = stream.addAttribute(CharTermAttribute.class);

            stream.reset();
            while (stream.incrementToken()) {
                Term term = new Term(DocumentBuilder.SENTENCE_FIELD, charTermAttribute.toString());
                output.add(new TermQuery(term), BooleanClause.Occur.SHOULD);
            }

            stream.end();
            stream.close();
        } catch (IOException e) {
            throw new Error("This should never happen", e);
        }
    }
}
