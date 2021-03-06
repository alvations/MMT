package eu.modernmt.lang;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

public class LanguageIndex {

    public static class Builder {

        private final Set<LanguagePair> languages = new HashSet<>();
        private final Map<LanguageKey, List<LanguageEntry>> index = new HashMap<>();
        private final Map<String, List<LanguageRule>> rules = new HashMap<>();

        public Builder add(LanguagePair pair) {
            LanguageKey key = LanguageKey.fromLanguage(pair);
            LanguageEntry entry = LanguageEntry.fromLanguage(pair);

            languages.add(pair);
            index.computeIfAbsent(key, k -> new ArrayList<>()).add(entry);

            return this;
        }

        public Builder addWildcardRule(Language lang, Language to) {
            return addRule(lang, null, to);
        }

        public Builder addRule(Language lang, Language from, Language to) {
            if (lang.getRegion() != null)
                throw new IllegalArgumentException("Language region not supported for rule: " + lang);

            rules.computeIfAbsent(lang.getLanguage(), k -> new ArrayList<>()).add(LanguageRule.make(from, to));

            return this;
        }

        public LanguageIndex build() {
            return new LanguageIndex(languages, index, rules);
        }

    }

    private final Set<LanguagePair> languages;
    private final Map<LanguageKey, List<LanguageEntry>> index;
    private final Map<String, List<LanguageRule>> rules;
    private final Set<Language> rulesSkipList;
    private final ConcurrentHashMap<LanguagePair, CacheEntry> mappingCache;

    protected LanguageIndex() {
        this.languages = null;
        this.index = null;
        this.rules = null;
        this.rulesSkipList = null;
        this.mappingCache = null;
    }

    private LanguageIndex(Set<LanguagePair> languages, Map<LanguageKey, List<LanguageEntry>> index, Map<String, List<LanguageRule>> rules) {
        this.languages = Collections.unmodifiableSet(languages);
        this.index = index;
        this.rules = rules;

        this.rulesSkipList = new HashSet<>();
        for (LanguagePair pair : languages) {
            if (pair.source.getRegion() != null)
                this.rulesSkipList.add(pair.source);
            if (pair.target.getRegion() != null)
                this.rulesSkipList.add(pair.target);
        }

        this.mappingCache = new ConcurrentHashMap<>();
    }

    public Set<LanguagePair> getLanguages() {
        return languages;
    }

    public int size() {
        return languages.size();
    }

    public LanguagePair asSingleLanguagePair() {
        return languages.size() == 1 ? languages.iterator().next() : null;
    }

    public LanguagePair mapIgnoringDirection(LanguagePair pair, boolean mask) {
        CacheEntry cached = mappingCache.get(pair);
        if (cached != null)
            return mask ? cached.masked : cached.normalized;
        cached = mappingCache.get(pair.reversed());
        if (cached != null)
            return (mask ? cached.masked : cached.normalized).reversed();

        LanguagePair mapped = map(pair, mask);

        if (mapped == null) {
            mapped = map(pair.reversed(), mask);
            if (mapped != null)
                mapped = mapped.reversed();
        }

        return mapped;
    }

    /**
     * Map the input language pair to one that is compatible with the supported ones,
     * trying to adapt language and region if necessary.
     * It does not try to map the reversed language pair, if needed call mapIgnoringDirection()
     *
     * @param pair the pair to search for
     * @param mask if true, forces the result to be one of the supported languages
     * @return the supported language pair that matches the input pair
     */
    public LanguagePair map(LanguagePair pair, boolean mask) {
        CacheEntry cached = mappingCache.computeIfAbsent(pair, this::search);
        if (cached == null)
            return null;
        return mask ? cached.masked : cached.normalized;
    }

    private CacheEntry search(LanguagePair language) {
        language = transform(language);

        LanguageKey key = LanguageKey.fromLanguage(language);
        List<LanguageEntry> entries = index.get(key);

        if (entries == null)
            return null;

        for (LanguageEntry entry : entries) {
            if (entry.match(language))
                return new CacheEntry(language, entry.getLanguagePair());
        }

        return null;
    }

    private LanguagePair transform(LanguagePair language) {
        Language source = transform(language.source);
        Language target = transform(language.target);

        if (source == null && target == null)
            return language;

        if (source == null)
            source = language.source;
        if (target == null)
            target = language.target;

        return new LanguagePair(source, target);
    }

    private Language transform(Language language) {
        if (rulesSkipList.contains(language))
            return null;

        List<LanguageRule> rules = this.rules.get(language.getLanguage());

        if (rules != null) {
            for (LanguageRule rule : rules) {
                if (rule.match(language))
                    return rule.getLanguage();
            }
        }

        return language.getRegion() == null ? null : new Language(language.getLanguage());
    }

    @Override
    public String toString() {
        return "i" + languages;
    }

    private static final class LanguageKey {

        private final String source;
        private final String target;

        public static LanguageKey fromLanguage(LanguagePair language) {
            return new LanguageKey(language.source.getLanguage(), language.target.getLanguage());
        }

        private LanguageKey(String source, String target) {
            this.source = source;
            this.target = target;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;

            LanguageKey that = (LanguageKey) o;

            if (!source.equals(that.source)) return false;
            return target.equals(that.target);
        }

        @Override
        public int hashCode() {
            int result = source.hashCode();
            result = 31 * result + target.hashCode();
            return result;
        }

        @Override
        public String toString() {
            return source + " > " + target;
        }
    }

    private static final class LanguageEntry {

        public static LanguageEntry fromLanguage(LanguagePair pair) {
            return new LanguageEntry(pair, Matcher.forLanguage(pair.source), Matcher.forLanguage(pair.target));
        }

        private final LanguagePair pair;
        private final Matcher source;
        private final Matcher target;

        private LanguageEntry(LanguagePair pair, Matcher source, Matcher target) {
            this.pair = pair;
            this.source = source;
            this.target = target;
        }

        public boolean match(LanguagePair pair) {
            return this.target.match(pair.target) && this.source.match(pair.source);
        }

        public LanguagePair getLanguagePair() {
            return pair;
        }

    }

    private static final class LanguageRule {

        public static LanguageRule make(Language from, Language to) {
            Matcher matcher = (from == null) ? Matcher.wildcardMatcher() : Matcher.exactMatcher(from);
            return new LanguageRule(matcher, to);
        }

        private final Matcher matcher;
        private final Language language;

        private LanguageRule(Matcher matcher, Language language) {
            this.matcher = matcher;
            this.language = language;
        }

        public boolean match(Language language) {
            return matcher.match(language);
        }

        public Language getLanguage() {
            return language;
        }

    }

    private interface Matcher {

        boolean match(Language language);

        static Matcher forLanguage(Language language) {
            return language.getRegion() == null ? wildcardMatcher() : exactMatcher(language);
        }

        static Matcher wildcardMatcher() {
            return language -> true;
        }

        static Matcher exactMatcher(Language language) {
            return language::equals;
        }

    }

    private static final class CacheEntry {

        public final LanguagePair normalized;
        public final LanguagePair masked;

        public CacheEntry(LanguagePair normalized, LanguagePair masked) {
            this.normalized = normalized;
            this.masked = masked;
        }

    }

}
