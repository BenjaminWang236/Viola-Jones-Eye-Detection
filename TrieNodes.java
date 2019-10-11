import java.util.*;
import java.util.stream.*;

// java.util.* and java.util.streams.* have been imported for this problem.
// You don't need any other imports.
public class TrieNodes {
    public static void main(String[] args) {
        Trie tmp = new Trie();
        tmp.insertWord("FIRES");
        System.out.println("Inserted");
        if (tmp.searchPrefix("F"))
            System.out.println("FOUND PREFIX");
        else
            System.out.println("FAILED PREFIX");
    }

    public static class TrieNode {
        Character c;
        Boolean isLeaf = false;
        HashMap<Character, TrieNode> children = new HashMap<>();

        public TrieNode() {
        }

        public TrieNode(Character c) {
            this.c = c;
        }
    }

    public static class Trie {
        private TrieNode root;

        // Implement these methods :
        public Trie() {
        }

        // public void printRootSimple() {
        // TrieNode current = root;
        // while()
        // }
        public void insertWord(String word) {
            if (word == null || word == "")
                return;
            char c0;
            int length = word.length();
            TrieNode current = root;
            for (int i = 0; i < length; i++) {
                if (current == null)
                    break;
                current.isLeaf = false;
                c0 = word.charAt(i);
                TrieNode next;
                if (!current.children.containsKey(c0)) {
                    System.out.println("Inserting " + c0);
                    next = new TrieNode(c0);
                    next.isLeaf = true;
                    current.children.put(c0, next);
                } else {
                    next = current.children.get(c0);
                }
                if (i + 1 == length)
                    break;
                current = next;
            }
        }

        public Boolean searchWord(String word) {
            if (word == null || word == "")
                return false;
            int length = word.length();
            char c0;
            TrieNode current = root;
            for (int i = 0; i < length; i++) {
                c0 = word.charAt(i);
                if (current == null || !current.children.containsKey(c0))
                    return false;
                current = current.children.get(c0);
                if (i + 1 == length && !current.isLeaf) {
                    return false;
                }
            }
            return true;
        }

        public Boolean searchPrefix(String word) {
            if (word == null || word == "")
                return false;
            int length = word.length();
            char c0;
            TrieNode current = root;
            for (int i = 0; i < length; i++) {
                c0 = word.charAt(i);
                // System.out.println(c0);
                if (current == null || !current.children.containsKey(c0))
                    return false;
                current = current.children.get(c0);
            }
            return true;
        }
    }

}
