# Design a key-value in-memory database that supports nested transactions. The database should support the following commands:
#
# SET <key> <value>: Sets the value for a key.
# GET <key>: Returns the value for a key.
# DELETE <key>: Removes a key.
# BEGIN: Starts a new transaction block.
# COMMIT: Commits all open transaction blocks, making the changes permanent.
# ROLLBACK: Reverts the changes in the most recent transaction block.
#
# The key challenge is handling the nesting. A COMMIT should only be final if it's the outermost transaction. A ROLLBACK should only undo the most recent transaction block, allowing the parent transaction to continue.
#
# ## Core Concepts & Rules
# The interviewer wants to see if you understand the rules of nested transactions:
# Isolation: Changes made inside a transaction are not visible outside of it until it's committed.
# Nested Visibility: A nested (child) transaction can see changes made by its parent transaction.
# Child Rollback: If a child transaction rolls back, its changes are discarded, but the parent transaction is unaffected and can continue.
# Parent Rollback: If a parent transaction rolls back, all changes from its children (even those that were "committed" to the parent) are also discarded.
# Hierarchical Commit: When a child transaction commits, its changes are passed to its parent transaction's scope. They only become permanent when the top-level, outermost transaction commits.
#
# ## High-Level Design & Data Structures
# A great way to solve this is by using a stack of hash maps (or dictionaries).
# Global Store: A single, primary hash map, let's call it datastore, holds all the permanently committed data.
# Transaction Stack: A stack, let's call it transaction_stack, where each element is a hash map representing a single transaction block.
# Here’s how the commands would work with this structure:

import unittest


# In-memory DB with nested transaction support
class InMemoryDB:
    def __init__(self):
        self.db = {}  # Main committed database (only updated on outermost commit)
        self.transactions = []  # Stack of transaction layers (dicts)

    def set(self, key, value):
        # If inside a transaction, only update the top transaction layer
        if self.transactions:
            self.transactions[-1][key] = value
        else:
            # If no transaction, update the main db directly
            self.db[key] = value

    def get(self, key):
        # Look for the key from the most recent transaction down to the main db
        for txn in reversed(self.transactions):
            if key in txn:
                # If key was unset, treat as missing
                return txn[key]
        # If not found in any transaction, check the main db
        return self.db.get(key, None)

    def unset(self, key):
        # Mark the key as deleted in the top transaction layer if inside a transaction
        if self.transactions:
            self.transactions[-1][key] = None
        else:
            # If no transaction, remove from main db
            self.db.pop(key, None)

    def begin(self):
        # Start a new transaction layer (nested transactions supported)
        self.transactions.append({})

    def rollback(self):
        # Discard the most recent transaction layer and all its changes
        if not self.transactions:
            print("NO TRANSACTION")
            return
        self.transactions.pop()

    def commit(self):
        # Merge the top transaction layer into its parent (or main db if outermost)
        if not self.transactions:
            print("NO TRANSACTION")
            return

        top_layer = self.transactions.pop()

        # If there are still transactions, merge into the next layer down
        if self.transactions:
            target = self.transactions[-1]
        else:
            # If no more transactions, merge into the main db
            target = self.db

        for key, value in top_layer.items():
            # Commit unset
            if value is None:
                target.pop(key, None)
            else:
                # Commit set
                target[key] = value


class TestInMemoryDB(unittest.TestCase):

    def setUp(self):
        """Create a new DB instance for each test."""
        self.db = InMemoryDB()

    def test_no_transaction(self):
        """Test basic operations without any transactions."""
        self.assertIsNone(self.db.get("a"))
        self.db.set("a", 10)
        self.assertEqual(self.db.get("a"), 10)
        self.db.unset("a")
        self.assertIsNone(self.db.get("a"))

    def test_single_transaction_commit(self):
        """Test a simple transaction that gets committed."""
        self.db.begin()
        self.db.set("a", 10)
        self.assertEqual(self.db.get("a"), 10)
        # The change should not be in the main db yet
        self.assertIsNone(self.db.db.get("a"))
        self.db.commit()
        # Now the change should be in the main db
        self.assertEqual(self.db.get("a"), 10)

    def test_single_transaction_rollback(self):
        """Test a simple transaction that gets rolled back."""
        self.db.set("a", 10)
        self.db.begin()
        self.db.set("a", 20)
        self.assertEqual(self.db.get("a"), 20)
        self.db.rollback()
        # The value should revert to its state before the transaction
        self.assertEqual(self.db.get("a"), 10)

    def test_nested_transaction_visibility(self):
        """Test that child transactions can see parent changes."""
        self.db.begin()  # Parent transaction
        self.db.set("a", 10)

        self.db.begin()  # Child transaction
        self.assertEqual(self.db.get("a"), 10)  # Child sees parent's 'a'
        self.db.set("b", 20)
        self.assertEqual(self.db.get("b"), 20)
        self.db.rollback()  # Rollback child

        # Parent should not see the child's change
        self.assertIsNone(self.db.get("b"))
        self.assertEqual(self.db.get("a"), 10)

    def test_nested_commit_and_final_commit(self):
        """Test committing a child and then the parent."""
        self.db.begin()  # T1
        self.db.set("a", 10)

        self.db.begin()  # T2
        self.db.set("b", 20)
        self.db.set("a", 30)  # Overwrite 'a'
        self.db.commit()  # Commit T2 into T1

        # We are back in T1, main db should still be empty
        self.assertDictEqual(self.db.db, {})
        # T1 should now have the merged changes
        self.assertEqual(self.db.get("a"), 30)
        self.assertEqual(self.db.get("b"), 20)

        self.db.commit()  # Final commit of T1

        # Main db should now have the final state
        self.assertEqual(self.db.get("a"), 30)
        self.assertEqual(self.db.get("b"), 20)

    def test_nested_commit_and_parent_rollback(self):
        """Test that a parent rollback discards a child's committed changes."""
        self.db.set("a", 5)
        self.db.begin()  # T1
        self.db.set("a", 10)

        self.db.begin()  # T2
        self.db.set("b", 20)
        self.db.commit()  # Commit T2 into T1

        # Back in T1, check that T2's changes are present
        self.assertEqual(self.db.get("b"), 20)

        # Now, rollback the parent transaction T1
        self.db.rollback()

        # All changes from T1 and committed changes from T2 should be gone
        self.assertIsNone(self.db.get("b"))
        self.assertEqual(self.db.get("a"), 5)  # Reverted to pre-transaction state

    def test_unset_in_transaction(self):
        """Test unsetting a key within transactions."""
        self.db.set("a", 10)
        self.db.set("b", 20)

        self.db.begin()  # T1
        self.db.set("a", 15)
        self.db.unset("b")

        self.assertEqual(self.db.get("a"), 15)
        self.assertIsNone(self.db.get("b"))  # 'b' appears unset inside T1

        self.db.begin()  # T2
        self.db.set("a", 25)
        self.db.set("b", 30)  # 'b' is re-set inside T2
        self.assertEqual(self.db.get("b"), 30)
        self.db.commit()  # Commit T2 into T1

        # Back in T1, changes from T2 are now visible
        self.assertEqual(self.db.get("a"), 25)
        self.assertEqual(self.db.get("b"), 30)

        self.db.rollback()  # Rollback T1

        # All transaction changes are gone, should be back to initial state
        self.assertEqual(self.db.get("a"), 10)
        self.assertEqual(self.db.get("b"), 20)


if __name__ == "__main__":
    unittest.main()
