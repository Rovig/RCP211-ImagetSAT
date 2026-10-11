#!/usr/bin/env python3
"""
Script pour exécuter tous les tests unitaires du projet
"""

import unittest
import sys

def run_all_tests():
    """Exécute tous les tests du projet"""
    # Découvrir et charger tous les tests
    loader = unittest.TestLoader()
    start_dir = 'tests'
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Exécuter les tests avec un runner verbeux
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Retourner 0 si succès, 1 sinon
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    sys.exit(run_all_tests())