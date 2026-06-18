"""
Topology Manager

Manages network topology selection and loading from text files.
Provides easy switching between different topologies.
"""

from typing import Dict, Any
import os


class TopologyManager:
    """Manages network topology selection and configuration."""
    
    AVAILABLE_TOPOLOGIES = {
        'dt12': 'dt12.txt',
        'test5': 'test5.txt',
    }
    
    @classmethod
    def load_topology(cls, topology_name: str) -> Dict[str, Any]:
        """
        Load a topology by name from a text file.
        
        Args:
            topology_name: Name of topology ('dt12' or 'test5')
            
        Returns:
            Dictionary containing topology data
            
        Raises:
            ValueError: If topology name is not recognized
            FileNotFoundError: If topology file doesn't exist
        """
        if topology_name.lower() not in cls.AVAILABLE_TOPOLOGIES:
            available = ', '.join(cls.AVAILABLE_TOPOLOGIES.keys())
            raise ValueError(
                f"Unknown topology '{topology_name}'. "
                f"Available topologies: {available}"
            )
        
        # Get the path to the topology file
        topology_file = cls.AVAILABLE_TOPOLOGIES[topology_name.lower()]
        topology_dir = os.path.dirname(os.path.abspath(__file__))
        topology_path = os.path.join(topology_dir, topology_file)
        
        # Read and parse the topology file
        return cls._parse_topology_file(topology_path)
    
    @classmethod
    def _parse_topology_file(cls, filepath: str) -> Dict[str, Any]:
        """
        Parse a topology text file.
        
        Args:
            filepath: Path to the topology file
            
        Returns:
            Dictionary containing topology data
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        data = {}
        current_section = None
        section_data = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Check for key=value pairs
            if '=' in line and not current_section:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                if key in ['NAME']:
                    data[key] = value
                elif key in ['N', 'LINKS']:
                    data[key] = int(value)
                continue
            
            # Check for section headers
            if line in ['TOPOLOGY', 'TOPOLOGY_LINK_LENGTHS', 'LINK_INDEX']:
                # Save previous section if exists
                if current_section:
                    data[current_section] = cls._parse_matrix(section_data)
                    section_data = []
                
                current_section = line
                continue
            
            # Collect section data
            if current_section:
                section_data.append(line)
        
        # Save last section
        if current_section and section_data:
            data[current_section] = cls._parse_matrix(section_data)
        
        return data
    
    @classmethod
    def _parse_matrix(cls, lines: list) -> list:
        """
        Parse a matrix from text lines.
        
        Args:
            lines: List of comma-separated values
            
        Returns:
            2D list (matrix)
        """
        matrix = []
        for line in lines:
            if line.strip():
                row = [int(x.strip()) for x in line.split(',')]
                matrix.append(row)
        return matrix
    
    @classmethod
    def list_topologies(cls) -> list:
        """List all available topology names."""
        return list(cls.AVAILABLE_TOPOLOGIES.keys())
