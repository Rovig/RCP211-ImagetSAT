#!/usr/bin/env python3
"""
Tests unitaires pour le système de logging
"""

import unittest
import logging
import tempfile
import os
from pathlib import Path

from utils.logger import setup_logger, get_logger, ColoredFormatter


class TestLogger(unittest.TestCase):
    """Tests pour le module logger"""
    
    def setUp(self):
        """Configuration avant chaque test"""
        # Nettoyer les loggers existants
        logger = logging.getLogger('TEST')
        logger.handlers = []
        logger.setLevel(logging.DEBUG)
    
    def test_setup_logger_basic(self):
        """Test de la création basique d'un logger"""
        logger = setup_logger('TEST', 'INFO')
        
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, 'TEST')
        self.assertEqual(logger.level, logging.INFO)
        self.assertTrue(len(logger.handlers) > 0)
    
    def test_setup_logger_with_file(self):
        """Test de la création d'un logger avec fichier"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_logger('TEST_FILE', 'DEBUG', 'test.log', tmpdir)
            
            # Vérifier qu'il y a 2 handlers (console + fichier)
            self.assertEqual(len(logger.handlers), 2)
            
            # Écrire un message
            logger.info("Test message")
            
            # Vérifier que le fichier existe
            log_files = list(Path(tmpdir).glob('*test.log'))
            self.assertTrue(len(log_files) > 0)
    
    def test_file_log_has_no_color_codes(self):
        """Test que les couleurs de la console n'atteignent pas le fichier"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_logger('TEST_NO_COLOR', 'DEBUG', 'test.log', tmpdir)
            
            logger.warning("Message d'avertissement")
            logger.error("Message d'erreur")
            for handler in logger.handlers:
                handler.flush()
            
            content = list(Path(tmpdir).glob('*test.log'))[0].read_text()
            
            self.assertIn("WARNING - Message d'avertissement", content)
            self.assertIn("ERROR - Message d'erreur", content)
            self.assertNotIn('\033[', content)
    
    def test_colored_formatter_does_not_modify_record(self):
        """Test que le formatter ne modifie pas le record partagé"""
        record = logging.LogRecord('test', logging.WARNING, __file__, 1, 'message', None, None)
        
        formatted = ColoredFormatter('%(levelname)s - %(message)s').format(record)
        
        self.assertIn('\033[', formatted)
        self.assertEqual(record.levelname, 'WARNING')
    
    def test_get_logger(self):
        """Test de récupération d'un logger"""
        logger1 = get_logger('TEST_GET')
        logger2 = get_logger('TEST_GET')
        
        # Les deux appels doivent retourner le même logger
        self.assertIs(logger1, logger2)
    
    def test_colored_formatter(self):
        """Test du formatter avec couleurs"""
        formatter = ColoredFormatter('%(levelname)s - %(message)s')
        
        # Créer un record de log
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='',
            lineno=0,
            msg='Test message',
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        
        # Vérifier que le format contient des codes couleur
        self.assertIn('\033[', formatted)  # Code ANSI
        self.assertIn('Test message', formatted)
    
    def test_logger_levels(self):
        """Test des différents niveaux de log"""
        logger = setup_logger('TEST_LEVELS', 'DEBUG')
        
        # Capturer les logs dans une liste
        test_handler = logging.Handler()
        records = []
        test_handler.emit = lambda record: records.append(record)
        logger.addHandler(test_handler)
        
        # Tester différents niveaux
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")
        
        # Vérifier que tous les messages sont capturés
        self.assertEqual(len(records), 5)
        # Les noms de niveau peuvent contenir des codes couleur ANSI
        self.assertIn('DEBUG', records[0].levelname)
        self.assertIn('INFO', records[1].levelname)
        self.assertIn('WARNING', records[2].levelname)
        self.assertIn('ERROR', records[3].levelname)
        self.assertIn('CRITICAL', records[4].levelname)


if __name__ == '__main__':
    unittest.main()