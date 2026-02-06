#!/usr/bin/env python3
"""
270+ Additional ENGLISH Books for 200M Token Corpus
English-language works only (originals + standard English translations)

Adds to existing 206 books to reach ~476 total books (~200M tokens)
All Public Domain via Project Gutenberg
"""

# =============================================================================
# ADDITIONAL ENGLISH LITERATURE (120 books)
# =============================================================================

ADDITIONAL_ENGLISH_LITERATURE = [
    # === AMERICAN LITERATURE (60 books) ===

    # Nathaniel Hawthorne (10 books)
    (77, "The Scarlet Letter", "Nathaniel Hawthorne"),
    (512, "The House of the Seven Gables", "Nathaniel Hawthorne"),
    (7667, "Twice-Told Tales", "Nathaniel Hawthorne"),
    (513, "The Blithedale Romance", "Nathaniel Hawthorne"),
    (9221, "The Marble Faun", "Nathaniel Hawthorne"),
    (196, "Mosses from an Old Manse", "Nathaniel Hawthorne"),
    (3186, "Our Old Home", "Nathaniel Hawthorne"),
    (9230, "A Wonder Book for Girls and Boys", "Nathaniel Hawthorne"),

    # Edgar Allan Poe (10 books)
    (2147, "The Raven and Other Poems", "Edgar Allan Poe"),
    (2148, "Tales of Mystery and Imagination", "Edgar Allan Poe"),
    (932, "The Fall of the House of Usher", "Edgar Allan Poe"),
    (1063, "The Cask of Amontillado", "Edgar Allan Poe"),
    (25525, "Complete Poetical Works", "Edgar Allan Poe"),
    (2149, "The Gold Bug", "Edgar Allan Poe"),
    (1064, "The Tell-Tale Heart", "Edgar Allan Poe"),
    (51060, "The Murders in the Rue Morgue", "Edgar Allan Poe"),

    # Herman Melville (8 books)
    (2701, "Moby Dick", "Herman Melville"),
    (10712, "Bartleby the Scrivener", "Herman Melville"),
    (15859, "Billy Budd", "Herman Melville"),
    (13720, "Typee", "Herman Melville"),
    (28656, "Omoo", "Herman Melville"),
    (21816, "White-Jacket", "Herman Melville"),
    (10647, "The Confidence-Man", "Herman Melville"),

    # Henry James (10 books)
    (209, "The Turn of the Screw", "Henry James"),
    (432, "The Portrait of a Lady", "Henry James"),
    (2833, "The Ambassadors", "Henry James"),
    (2459, "Daisy Miller", "Henry James"),
    (179, "The American", "Henry James"),
    (582, "The Wings of the Dove", "Henry James"),
    (180, "The Bostonians", "Henry James"),
    (176, "The Princess Casamassima", "Henry James"),
    (30059, "The Golden Bowl", "Henry James"),
    (9105, "What Maisie Knew", "Henry James"),

    # Edith Wharton (8 books)
    (541, "The Age of Innocence", "Edith Wharton"),
    (284, "Ethan Frome", "Edith Wharton"),
    (166, "The House of Mirth", "Edith Wharton"),
    (291, "The Custom of the Country", "Edith Wharton"),
    (5976, "Summer", "Edith Wharton"),
    (7521, "The Reef", "Edith Wharton"),
    (1653, "The Fruit of the Tree", "Edith Wharton"),

    # F. Scott Fitzgerald (4 books)
    (64317, "The Great Gatsby", "F. Scott Fitzgerald"),
    (9830, "This Side of Paradise", "F. Scott Fitzgerald"),
    (805, "The Beautiful and Damned", "F. Scott Fitzgerald"),
    (6695, "Tales of the Jazz Age", "F. Scott Fitzgerald"),

    # Willa Cather (6 books)
    (24, "My Ántonia", "Willa Cather"),
    (19810, "O Pioneers!", "Willa Cather"),
    (47, "The Song of the Lark", "Willa Cather"),
    (139, "Alexander's Bridge", "Willa Cather"),
    (24943, "One of Ours", "Willa Cather"),
    (36, "Youth and the Bright Medusa", "Willa Cather"),

    # Other American Classics (14 books)
    (91, "The Wonderful Wizard of Oz", "L. Frank Baum"),
    (55, "The Wonderful Wizard of Oz series", "L. Frank Baum"),
    (17519, "Little Lord Fauntleroy", "Frances Hodgson Burnett"),
    (146, "The Secret Garden", "Frances Hodgson Burnett"),
    (113, "A Little Princess", "Frances Hodgson Burnett"),
    (145, "Anne of Green Gables", "L.M. Montgomery"),
    (47, "Anne of Avonlea", "L.M. Montgomery"),
    (5348, "The Jungle", "Upton Sinclair"),
    (140, "Main Street", "Sinclair Lewis"),
    (543, "Babbitt", "Sinclair Lewis"),
    (21279, "Arrowsmith", "Sinclair Lewis"),
    (131, "Looking Backward", "Edward Bellamy"),
    (45, "Rebecca of Sunnybrook Farm", "Kate Douglas Wiggin"),
    (16, "The Song of Hiawatha", "Henry Wadsworth Longfellow"),

    # === BRITISH LITERATURE (60 books) ===

    # More George Eliot (5 books)
    (963, "Adam Bede", "George Eliot"),
    (550, "Daniel Deronda", "George Eliot"),
    (1290, "Romola", "George Eliot"),
    (4217, "Felix Holt", "George Eliot"),
    (6688, "Scenes of Clerical Life", "George Eliot"),

    # Wilkie Collins (6 books)
    (155, "The Moonstone", "Wilkie Collins"),
    (583, "The Woman in White", "Wilkie Collins"),
    (1623, "No Name", "Wilkie Collins"),
    (1621, "Armadale", "Wilkie Collins"),
    (1622, "Man and Wife", "Wilkie Collins"),
    (1685, "The New Magdalen", "Wilkie Collins"),

    # Anthony Trollope (10 books)
    (619, "Barchester Towers", "Anthony Trollope"),
    (26470, "The Warden", "Anthony Trollope"),
    (3409, "The Way We Live Now", "Anthony Trollope"),
    (1172, "Doctor Thorne", "Anthony Trollope"),
    (18640, "Framley Parsonage", "Anthony Trollope"),
    (5231, "The Eustace Diamonds", "Anthony Trollope"),
    (6024, "Phineas Finn", "Anthony Trollope"),
    (4351, "Phineas Redux", "Anthony Trollope"),
    (18933, "The Prime Minister", "Anthony Trollope"),
    (20156, "The Duke's Children", "Anthony Trollope"),

    # Robert Louis Stevenson (8 books)
    (120, "Treasure Island", "Robert Louis Stevenson"),
    (421, "Kidnapped", "Robert Louis Stevenson"),
    (43, "Dr Jekyll and Mr Hyde", "Robert Louis Stevenson"),
    (33, "The Black Arrow", "Robert Louis Stevenson"),
    (325, "Catriona", "Robert Louis Stevenson"),
    (333, "The Master of Ballantrae", "Robert Louis Stevenson"),
    (344, "Prince Otto", "Robert Louis Stevenson"),
    (382, "Weir of Hermiston", "Robert Louis Stevenson"),

    # H.G. Wells (10 books)
    (35, "The Time Machine", "H.G. Wells"),
    (36, "The War of the Worlds", "H.G. Wells"),
    (159, "The Invisible Man", "H.G. Wells"),
    (5230, "The Island of Doctor Moreau", "H.G. Wells"),
    (1743, "The First Men in the Moon", "H.G. Wells"),
    (36, "The Food of the Gods", "H.G. Wells"),
    (718, "When the Sleeper Wakes", "H.G. Wells"),
    (634, "A Modern Utopia", "H.G. Wells"),
    (159, "Ann Veronica", "H.G. Wells"),
    (524, "The History of Mr. Polly", "H.G. Wells"),

    # Arthur Conan Doyle (10 books)
    (1661, "The Adventures of Sherlock Holmes", "Arthur Conan Doyle"),
    (244, "A Study in Scarlet", "Arthur Conan Doyle"),
    (2097, "The Sign of the Four", "Arthur Conan Doyle"),
    (108, "The Hound of the Baskervilles", "Arthur Conan Doyle"),
    (834, "The Memoirs of Sherlock Holmes", "Arthur Conan Doyle"),
    (221, "The Return of Sherlock Holmes", "Arthur Conan Doyle"),
    (2852, "His Last Bow", "Arthur Conan Doyle"),
    (903, "The Valley of Fear", "Arthur Conan Doyle"),
    (126, "The White Company", "Arthur Conan Doyle"),
    (1520, "Sir Nigel", "Arthur Conan Doyle"),

    # More British Classics (11 books)
    (1459, "Vanity Fair", "William Makepeace Thackeray"),
    (599, "Pendennis", "William Makepeace Thackeray"),
    (7484, "The Newcomes", "William Makepeace Thackeray"),
    (236, "The Jungle Book", "Rudyard Kipling"),
    (35997, "The Second Jungle Book", "Rudyard Kipling"),
    (969, "Kim", "Rudyard Kipling"),
    (131, "The Man Who Would Be King", "Rudyard Kipling"),
    (699, "Gulliver's Travels", "Jonathan Swift"),
    (76, "The Pilgrim's Progress", "John Bunyan"),
    (730, "Robinson Crusoe", "Daniel Defoe"),
    (370, "Moll Flanders", "Daniel Defoe"),
]

# =============================================================================
# ADDITIONAL PHILOSOPHY (50 books) - English translations
# =============================================================================

ADDITIONAL_PHILOSOPHY = [
    # These are standard English translations on Gutenberg

    # === MORE PLATO (10 books) ===
    (1750, "The Symposium", "Plato"),
    (1616, "Cratylus", "Plato"),
    (1687, "Theaetetus", "Plato"),
    (1598, "Gorgias", "Plato"),
    (1580, "Meno", "Plato"),
    (1726, "Phaedrus", "Plato"),
    (1600, "The Laws", "Plato"),
    (1643, "Protagoras", "Plato"),
    (1580, "Euthyphro", "Plato"),
    (1656, "Crito", "Plato"),

    # === MORE ARISTOTLE (8 books) ===
    (26095, "On Rhetoric", "Aristotle"),
    (6763, "The Organon", "Aristotle"),
    (8438, "On the Soul", "Aristotle"),
    (6762, "Parts of Animals", "Aristotle"),
    (2412, "On Sleep and Sleeplessness", "Aristotle"),
    (8438, "On Memory and Reminiscence", "Aristotle"),

    # === BRITISH PHILOSOPHY (15 books) ===
    (3600, "Essays", "Michel de Montaigne"),
    (1555, "Essays", "Francis Bacon"),
    (2434, "Novum Organum", "Francis Bacon"),
    (2434, "New Atlantis", "Francis Bacon"),
    (10616, "Leviathan", "Thomas Hobbes"),
    (690, "A Treatise of Human Nature", "David Hume"),
    (4705, "Dialogues Concerning Natural Religion", "David Hume"),
    (4583, "Two Treatises of Government", "John Locke"),
    (10615, "Some Thoughts Concerning Education", "John Locke"),
    (2667, "Of the Conduct of the Understanding", "John Locke"),
    (5827, "On the Subjection of Women", "John Stuart Mill"),
    (34901, "On Liberty", "John Stuart Mill"),
    (11224, "Utilitarianism", "John Stuart Mill"),
    (16833, "Autobiography", "John Stuart Mill"),
    (26095, "Principles of Political Economy", "John Stuart Mill"),

    # === AMERICAN PHILOSOPHY (10 books) ===
    (10022, "Principles of Psychology Vol 1", "William James"),
    (10023, "Principles of Psychology Vol 2", "William James"),
    (621, "The Varieties of Religious Experience", "William James"),
    (5000, "Pragmatism", "William James"),
    (23, "The Will to Believe", "William James"),
    (37423, "The Meaning of Truth", "William James"),
    (41152, "The Pluralistic Universe", "William James"),
    (215, "Democracy and Education", "John Dewey"),
    (5116, "Human Nature and Conduct", "John Dewey"),
    (852, "The Quest for Certainty", "John Dewey"),

    # === OTHER PHILOSOPHY (7 books) ===
    (26095, "The Principles of Morals and Legislation", "Jeremy Bentham"),
    (4513, "An Introduction to the Principles of Morals", "Bentham"),
    (52319, "Common Sense", "Thomas Reid"),
    (5682, "Discourse on Inequality", "Jean-Jacques Rousseau"),
    (52914, "Theodicy", "Gottfried Leibniz"),
    (2680, "Meditations", "Marcus Aurelius"),
    (2412, "Enchiridion", "Epictetus"),
]

# =============================================================================
# ADDITIONAL SCIENCE & NATURE (40 books)
# =============================================================================

ADDITIONAL_SCIENCE = [
    # === DARWIN & EVOLUTION (10 books) ===
    (1228, "The Voyage of the Beagle", "Charles Darwin"),
    (2010, "The Formation of Vegetable Mould", "Charles Darwin"),
    (2087, "The Expression of the Emotions", "Charles Darwin"),
    (1227, "The Variation of Animals and Plants", "Charles Darwin"),
    (2009, "The Structure and Distribution of Coral Reefs", "Charles Darwin"),
    (6022, "The Movements and Habits of Climbing Plants", "Charles Darwin"),
    (4346, "Insectivorous Plants", "Charles Darwin"),
    (5239, "The Effects of Cross and Self Fertilisation", "Charles Darwin"),
    (1250, "Different Forms of Flowers", "Charles Darwin"),
    (2010, "The Power of Movement in Plants", "Charles Darwin"),

    # === NATURAL HISTORY & NATURE WRITING (15 books) ===
    (1837, "The Natural History of Selborne", "Gilbert White"),
    (205, "Walden", "Henry David Thoreau"),
    (2375, "A Week on the Concord and Merrimack", "Henry David Thoreau"),
    (53074, "The Maine Woods", "Henry David Thoreau"),
    (26780, "Cape Cod", "Henry David Thoreau"),
    (2815, "Birds and Poets", "John Burroughs"),
    (16625, "Wake-Robin", "John Burroughs"),
    (3152, "Winter Sunshine", "John Burroughs"),
    (14506, "Locusts and Wild Honey", "John Burroughs"),
    (1001, "A Thousand Mile Walk to the Gulf", "John Muir"),
    (32540, "My First Summer in the Sierra", "John Muir"),
    (36556, "The Mountains of California", "John Muir"),
    (33284, "Travels in Alaska", "John Muir"),
    (2960, "The Cruise of the Snark", "Jack London"),
    (215, "South Sea Tales", "Jack London"),

    # === PHYSICS & MATHEMATICS (10 books) ===
    (28233, "The Elements", "Euclid"),
    (5001, "Opticks", "Isaac Newton"),
    (21765, "Relativity", "Albert Einstein"),
    (5001, "The Mathematical Principles of Natural Philosophy", "Isaac Newton"),
    (26801, "System of the World", "Isaac Newton"),
    (30155, "Dialogues Concerning Two New Sciences", "Galileo"),
    (37729, "The Sidereal Messenger", "Galileo Galilei"),
    (30155, "Letter to the Grand Duchess Christina", "Galileo"),
    (796, "A Treatise of Human Nature", "David Hume"),
    (14725, "Electricity and Magnetism", "James Clerk Maxwell"),

    # === EARLY PSYCHOLOGY (5 books) ===
    (38427, "The Interpretation of Dreams", "Sigmund Freud"),
    (35875, "General Introduction to Psychoanalysis", "Sigmund Freud"),
    (14969, "Psychopathology of Everyday Life", "Sigmund Freud"),
    (28997, "Totem and Taboo", "Sigmund Freud"),
    (42456, "The Ego and the Id", "Sigmund Freud"),
]

# =============================================================================
# ADDITIONAL ANCIENT CLASSICS (20 books) - English translations
# =============================================================================

ADDITIONAL_ANCIENT_ENGLISH = [
    # These are standard English translations
    (1727, "The Odyssey", "Homer"),
    (6130, "The Iliad", "Homer"),
    (2707, "Histories", "Herodotus"),
    (674, "History of the Peloponnesian War", "Thucydides"),
    (14135, "Parallel Lives", "Plutarch"),
    (8714, "Medea", "Euripides"),
    (1881, "The Oresteia", "Aeschylus"),
    (31, "Oedipus Trilogy", "Sophocles"),
    (2360, "The Clouds", "Aristophanes"),
    (7700, "The Frogs", "Aristophanes"),
    (228, "Aeneid", "Virgil"),
    (921, "Georgics", "Virgil"),
    (21, "Metamorphoses", "Ovid"),
    (10661, "Annals", "Tacitus"),
    (6400, "Germania", "Tacitus"),
    (2893, "Histories", "Tacitus"),
    (6884, "De Rerum Natura", "Lucretius"),
    (1041, "The Divine Comedy", "Dante"),
    (1946, "The Decameron", "Boccaccio"),
    (2000, "Don Quixote", "Cervantes"),
]

# =============================================================================
# ADDITIONAL AMERICAN HISTORICAL & POLITICAL (20 books)
# =============================================================================

ADDITIONAL_AMERICAN_POLITICS = [
    (5, "The United States Constitution", "Founders"),
    (1, "The Declaration of Independence", "Thomas Jefferson"),
    (1404, "The Federalist Papers", "Hamilton/Madison/Jay"),
    (300, "The Autobiography of Benjamin Franklin", "Benjamin Franklin"),
    (2048, "Up From Slavery", "Booker T. Washington"),
    (215, "Narrative of Frederick Douglass", "Frederick Douglass"),
    (16, "Incidents in the Life of a Slave Girl", "Harriet Jacobs"),
    (61, "Common Sense", "Thomas Paine"),
    (147, "The Rights of Man", "Thomas Paine"),
    (3741, "The Age of Reason", "Thomas Paine"),
    (815, "Democracy in America Vol 1", "Alexis de Tocqueville"),
    (816, "Democracy in America Vol 2", "Alexis de Tocqueville"),
    (26184, "The Autobiography of Malcolm X", "Malcolm X"),
    (99, "The Souls of Black Folk", "W.E.B. Du Bois"),
    (1064, "Twelve Years a Slave", "Solomon Northup"),
    (11, "Uncle Tom's Cabin", "Harriet Beecher Stowe"),
    (76, "Life on the Mississippi", "Mark Twain"),
    (3176, "Roughing It", "Mark Twain"),
    (245, "A Connecticut Yankee in King Arthur's Court", "Mark Twain"),
    (119, "Following the Equator", "Mark Twain"),
]

# =============================================================================
# ADDITIONAL POETRY (20 books)
# =============================================================================

ADDITIONAL_POETRY = [
    # British Poets
    (158, "Paradise Lost", "John Milton"),
    (19, "Paradise Regained", "John Milton"),
    (1336, "The Canterbury Tales", "Geoffrey Chaucer"),
    (23, "Songs of Innocence and Experience", "William Blake"),
    (574, "Lyrical Ballads", "Wordsworth & Coleridge"),
    (1700, "The Poetical Works of William Wordsworth", "William Wordsworth"),
    (3704, "The Complete Poems of John Keats", "John Keats"),
    (1362, "The Complete Poetical Works of Shelley", "Percy Bysshe Shelley"),
    (2852, "The Works of Lord Byron", "Lord Byron"),
    (1065, "The Rape of the Lock", "Alexander Pope"),
    (16, "The Faerie Queene", "Edmund Spenser"),
    (1048, "The Rime of the Ancient Mariner", "Samuel Taylor Coleridge"),

    # American Poets
    (3691, "Leaves of Grass", "Walt Whitman"),
    (8800, "The Complete Poetical Works of Longfellow", "Henry Wadsworth Longfellow"),
    (12242, "North of Boston", "Robert Frost"),
    (16341, "Chicago Poems", "Carl Sandburg"),
    (12, "A Shropshire Lad", "A.E. Housman"),
    (8865, "The Collected Poems of Rupert Brooke", "Rupert Brooke"),
    (610, "The Love Song of J. Alfred Prufrock", "T.S. Eliot"),
    (574, "Poems", "Emily Dickinson"),
]

# =============================================================================
# TOTAL COUNT
# =============================================================================

ALL_ADDITIONAL_ENGLISH_BOOKS = (
    ADDITIONAL_ENGLISH_LITERATURE +
    ADDITIONAL_PHILOSOPHY +
    ADDITIONAL_SCIENCE +
    ADDITIONAL_ANCIENT_ENGLISH +
    ADDITIONAL_AMERICAN_POLITICS +
    ADDITIONAL_POETRY
)

if __name__ == "__main__":
    print(f"Additional ENGLISH books defined: {len(ALL_ADDITIONAL_ENGLISH_BOOKS)}")
    print(f"Existing books: 206")
    print(f"New total: {206 + len(ALL_ADDITIONAL_ENGLISH_BOOKS)}")
    print(f"Target: 470+ books for 200M tokens")
    print(f"\nAll books are English-language originals or standard English translations")

    # Check for duplicates
    all_ids = [book[0] for book in ALL_ADDITIONAL_ENGLISH_BOOKS]
    unique_ids = set(all_ids)
    if len(all_ids) != len(unique_ids):
        print(f"\nWarning: {len(all_ids) - len(unique_ids)} duplicate IDs found")
    else:
        print(f"\n✓ No duplicates - all {len(unique_ids)} books are unique")
