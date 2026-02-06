#!/usr/bin/env python3
"""
K-SSM Corpus Expansion to ~60M Tokens
Merges original 206 books with 92 new English books
Total: ~298 unique books, ~59M tokens
"""

# Original books from build_corpus_200m.py
from build_corpus_200m import (
    GUTENBERG_LITERATURE,
    PHILOSOPHY_BOOKS,
    RELIGIOUS_BOOKS,
    SCIENCE_BOOKS,
    POLITICAL_BOOKS,
    ESSAY_BOOKS,
    ANCIENT_BOOKS
)

# =============================================================================
# ADDITIONAL 92 NEW ENGLISH BOOKS
# =============================================================================

NEW_ENGLISH_BOOKS = [
    (16, "The Faerie Queene", "Edmund Spenser"),
    (16, "Incidents in the Life of a Slave Girl", "Harriet Jacobs"),
    (23, "Songs of Innocence and Experience", "William Blake"),
    (24, "My Ántonia", "Willa Cather"),
    (28, "The Call of the Wild", "Jack London"),
    (30, "The King James Bible Complete", "Various"),
    (77, "The Scarlet Letter", "Nathaniel Hawthorne"),
    (91, "The Wonderful Wizard of Oz", "L. Frank Baum"),
    (147, "The Rights of Man", "Thomas Paine"),
    (155, "The Moonstone", "Wilkie Collins"),
    (166, "The House of Mirth", "Edith Wharton"),
    (179, "The American", "Henry James"),
    (284, "Ethan Frome", "Edith Wharton"),
    (300, "The Autobiography of Benjamin Franklin", "Benjamin Franklin"),
    (421, "Kidnapped", "Robert Louis Stevenson"),
    (512, "The House of the Seven Gables", "Nathaniel Hawthorne"),
    (541, "The Age of Innocence", "Edith Wharton"),
    (550, "Daniel Deronda", "George Eliot"),
    (574, "Lyrical Ballads", "Wordsworth & Coleridge"),
    (583, "The Woman in White", "Wilkie Collins"),
    (619, "Barchester Towers", "Anthony Trollope"),
    (621, "The Exemplary Novels", "Miguel de Cervantes"),
    (621, "The Varieties of Religious Experience", "William James"),
    (690, "A Treatise of Human Nature", "David Hume"),
    (816, "Democracy in America Vol 2", "Alexis de Tocqueville"),
    (848, "Poems", "Emily Dickinson"),
    (921, "Georgics", "Virgil"),
    (932, "The Fall of the House of Usher", "Edgar Allan Poe"),
    (963, "Adam Bede", "George Eliot"),
    (1041, "The Divine Comedy", "Dante Alighieri"),
    (1044, "The Scarlet Pimpernel", "Baroness Orczy"),
    (1063, "The Cask of Amontillado", "Edgar Allan Poe"),
    (1065, "The Rape of the Lock", "Alexander Pope"),
    (1074, "Walden", "Henry David Thoreau"),
    (1227, "The Variation of Animals and Plants", "Charles Darwin"),
    (1290, "Romola", "George Eliot"),
    (1336, "The Canterbury Tales", "Geoffrey Chaucer"),
    (1362, "Poems", "Percy Bysshe Shelley"),
    (1580, "Meno", "Plato"),
    (1598, "Gorgias", "Plato"),
    (1687, "Theaetetus", "Plato"),
    (1700, "Poems", "William Wordsworth"),
    (1726, "Phaedrus", "Plato"),
    (1743, "The First Men in the Moon", "H.G. Wells"),
    (1837, "The Natural History of Selborne", "Gilbert White"),
    (1881, "The Oresteia", "Aeschylus"),
    (1946, "The Decameron", "Giovanni Boccaccio"),
    (1998, "The Birth of Tragedy", "Friedrich Nietzsche"),
    (2000, "Don Quixote", "Miguel de Cervantes"),
    (2010, "The Formation of Vegetable Mould", "Charles Darwin"),
    (2048, "Up From Slavery", "Booker T. Washington"),
    (2087, "The Expression of the Emotions", "Charles Darwin"),
    (2097, "The Sign of the Four", "Arthur Conan Doyle"),
    (2147, "The Raven", "Edgar Allan Poe"),
    (2147, "Confessions", "Jean-Jacques Rousseau"),
    (2211, "A Sportsman's Sketches", "Ivan Turgenev"),
    (2346, "The I Ching", "Various"),
    (2360, "The Clouds", "Aristophanes"),
    (2375, "A Week on the Concord", "Henry David Thoreau"),
    (2459, "Daisy Miller", "Henry James"),
    (2815, "Birds and Poets", "John Burroughs"),
    (2833, "The Ambassadors", "Henry James"),
    (2852, "Poems", "Lord Byron"),
    (2893, "Histories", "Tacitus"),
    (2958, "Don Quixote Vol 2", "Miguel de Cervantes"),
    (3065, "The Art of Love", "Ovid"),
    (3409, "The Way We Live Now", "Anthony Trollope"),
    (3691, "Leaves of Grass", "Walt Whitman"),
    (3704, "The Complete Poems", "John Keats"),
    (3726, "Orlando Furioso", "Ludovico Ariosto"),
    (3913, "Emile", "Jean-Jacques Rousseau"),
    (4583, "Two Treatises of Government", "John Locke"),
    (6400, "Germania", "Tacitus"),
    (6884, "De Rerum Natura", "Lucretius"),
    (7667, "Twice-Told Tales", "Nathaniel Hawthorne"),
    (7700, "The Frogs", "Aristophanes"),
    (8413, "On the Eve", "Ivan Turgenev"),
    (8800, "The Complete Poems", "Henry Wadsworth Longfellow"),
    (9830, "This Side of Paradise", "F. Scott Fitzgerald"),
    (10615, "Some Thoughts Concerning Education", "John Locke"),
    (10661, "Annals", "Tacitus"),
    (10712, "Bartleby the Scrivener", "Herman Melville"),
    (12242, "Poems", "Robert Frost"),
    (13415, "The Chorus Girl", "Anton Chekhov"),
    (14420, "La Celestina", "Fernando de Rojas"),
    (15859, "Billy Budd", "Herman Melville"),
    (16341, "Chicago Poems", "Carl Sandburg"),
    (16625, "Wake-Robin", "John Burroughs"),
    (19810, "O Pioneers!", "Willa Cather"),
    (25525, "Complete Poetical Works", "Edgar Allan Poe"),
    (26470, "The Warden", "Anthony Trollope"),
    (37729, "Dialogue Concerning Two New Sciences", "Galileo"),
    (52319, "Twilight of the Idols", "Friedrich Nietzsche"),
    (53074, "The Maine Woods", "Henry David Thoreau"),
    (64317, "The Great Gatsby", "F. Scott Fitzgerald"),

]

# =============================================================================
# COMBINED CORPUS
# =============================================================================

ALL_BOOKS = (
    GUTENBERG_LITERATURE +
    PHILOSOPHY_BOOKS +
    RELIGIOUS_BOOKS +
    SCIENCE_BOOKS +
    POLITICAL_BOOKS +
    ESSAY_BOOKS +
    ANCIENT_BOOKS +
    NEW_ENGLISH_BOOKS
)

if __name__ == "__main__":
    book_ids = [b[0] for b in ALL_BOOKS]
    unique_ids = set(book_ids)
    
    print(f"Total books: {len(ALL_BOOKS)}")
    print(f"Unique book IDs: {len(unique_ids)}")
    print(f"Duplicates: {len(book_ids) - len(unique_ids)}")
    print(f"\nEstimated tokens: ~{len(unique_ids) * 199_000:,} (~{len(unique_ids) * 199_000 / 1e6:.1f}M)")
