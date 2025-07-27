"""
Calculator Plugin for Jarvis v2.0
Provides mathematical calculation capabilities
"""

import math
import ast
import operator
from typing import List, Dict, Any
from plugin_system import CommandPlugin, PluginMetadata, PluginType, PluginPriority, PluginContext, PluginResponse, PluginConfig

class CalculatorPlugin(CommandPlugin):
    """Mathematical calculator plugin"""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="calculator",
            version="1.0.0",
            description="Perform mathematical calculations and conversions",
            author="Jarvis Team",
            plugin_type=PluginType.COMMAND,
            priority=PluginPriority.NORMAL,
            dependencies=[],
            permissions=[],
            config_schema={
                "precision": {
                    "type": "integer",
                    "description": "Decimal precision for results",
                    "default": 6
                },
                "max_expression_length": {
                    "type": "integer",
                    "description": "Maximum allowed expression length",
                    "default": 1000
                }
            },
            tags=["math", "calculator", "utility", "computation"]
        )
    
    def __init__(self, plugin_manager):
        super().__init__(plugin_manager)
        
        # Safe operators for expression evaluation
        self.operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.Mod: operator.mod,
            ast.FloorDiv: operator.floordiv,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }
        
        # Safe functions
        self.functions = {
            'abs': abs,
            'round': round,
            'max': max,
            'min': min,
            'sum': sum,
            'sqrt': math.sqrt,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'asin': math.asin,
            'acos': math.acos,
            'atan': math.atan,
            'log': math.log,
            'log10': math.log10,
            'exp': math.exp,
            'ceil': math.ceil,
            'floor': math.floor,
            'factorial': math.factorial,
            'degrees': math.degrees,
            'radians': math.radians,
            'pi': math.pi,
            'e': math.e
        }
    
    async def initialize(self, config: PluginConfig) -> bool:
        """Initialize calculator plugin"""
        try:
            self.config = config
            self.precision = self.get_config_value("precision", 6)
            self.max_expression_length = self.get_config_value("max_expression_length", 1000)
            
            self.logger.info("Calculator plugin initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize calculator plugin: {e}")
            return False
    
    def get_commands(self) -> List[str]:
        return ["calc", "calculate", "math", "convert"]
    
    async def handle_command(self, command: str, args: List[str], context: PluginContext) -> PluginResponse:
        """Handle calculator commands"""
        try:
            if command in ["calc", "calculate", "math"]:
                return await self.handle_calculation(args, context)
            elif command == "convert":
                return await self.handle_conversion(args, context)
            
            return PluginResponse(
                success=False,
                error=f"Unknown calculator command: {command}"
            )
            
        except Exception as e:
            self.logger.error(f"Error in calculator command: {e}")
            return PluginResponse(
                success=False,
                error=f"Calculator error: {str(e)}"
            )
    
    async def handle_calculation(self, args: List[str], context: PluginContext) -> PluginResponse:
        """Handle mathematical calculation"""
        if not args:
            return PluginResponse(
                success=False,
                error="Please provide a mathematical expression to calculate"
            )
        
        expression = " ".join(args)
        
        if len(expression) > self.max_expression_length:
            return PluginResponse(
                success=False,
                error=f"Expression too long (max {self.max_expression_length} characters)"
            )
        
        try:
            result = self.safe_eval(expression)
            
            # Format result
            if isinstance(result, float):
                if result.is_integer():
                    result_str = str(int(result))
                else:
                    result_str = f"{result:.{self.precision}g}"
            else:
                result_str = str(result)
            
            response_text = f"🧮 **Calculator**\\n\\n"
            response_text += f"**Expression:** `{expression}`\\n"
            response_text += f"**Result:** `{result_str}`"
            
            return PluginResponse(
                success=True,
                result=response_text,
                data={
                    "expression": expression,
                    "result": result,
                    "result_string": result_str,
                    "result_type": type(result).__name__
                },
                should_continue=False
            )
            
        except Exception as e:
            return PluginResponse(
                success=False,
                error=f"Calculation error: {str(e)}"
            )
    
    async def handle_conversion(self, args: List[str], context: PluginContext) -> PluginResponse:
        """Handle unit conversions"""
        if len(args) < 3:
            return PluginResponse(
                success=False,
                error="Usage: /convert <value> <from_unit> <to_unit>"
            )
        
        try:
            value = float(args[0])
            from_unit = args[1].lower()
            to_unit = args[2].lower()
            
            result = self.convert_units(value, from_unit, to_unit)
            
            response_text = f"🔄 **Unit Conversion**\\n\\n"
            response_text += f"**{value} {from_unit}** = **{result:.{self.precision}g} {to_unit}**"
            
            return PluginResponse(
                success=True,
                result=response_text,
                data={
                    "original_value": value,
                    "from_unit": from_unit,
                    "to_unit": to_unit,
                    "result": result
                },
                should_continue=False
            )
            
        except ValueError as e:
            return PluginResponse(
                success=False,
                error=f"Invalid value: {args[0]}"
            )
        except Exception as e:
            return PluginResponse(
                success=False,
                error=f"Conversion error: {str(e)}"
            )
    
    def safe_eval(self, expression: str):
        """Safely evaluate mathematical expression"""
        # Replace common math constants and functions
        expression = expression.replace('^', '**')  # Power operator
        
        # Parse the expression
        try:
            node = ast.parse(expression, mode='eval')
        except SyntaxError:
            raise ValueError("Invalid mathematical expression")
        
        return self._eval_node(node.body)
    
    def _eval_node(self, node):
        """Recursively evaluate AST nodes"""
        if isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
        elif isinstance(node, ast.Num):  # Python < 3.8
            return node.n
        elif isinstance(node, ast.Name):
            if node.id in self.functions:
                return self.functions[node.id]
            else:
                raise ValueError(f"Unknown variable or function: {node.id}")
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op = self.operators.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operation: {type(node.op).__name__}")
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op = self.operators.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operation: {type(node.op).__name__}")
            return op(operand)
        elif isinstance(node, ast.Call):
            func = self._eval_node(node.func)
            if not callable(func):
                raise ValueError(f"Not a function: {func}")
            args = [self._eval_node(arg) for arg in node.args]
            return func(*args)
        elif isinstance(node, ast.List):
            return [self._eval_node(item) for item in node.elts]
        elif isinstance(node, ast.Tuple):
            return tuple(self._eval_node(item) for item in node.elts)
        else:
            raise ValueError(f"Unsupported expression type: {type(node).__name__}")
    
    def convert_units(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert between different units"""
        
        # Temperature conversions
        if from_unit in ['c', 'celsius'] and to_unit in ['f', 'fahrenheit']:
            return (value * 9/5) + 32
        elif from_unit in ['f', 'fahrenheit'] and to_unit in ['c', 'celsius']:
            return (value - 32) * 5/9
        elif from_unit in ['c', 'celsius'] and to_unit in ['k', 'kelvin']:
            return value + 273.15
        elif from_unit in ['k', 'kelvin'] and to_unit in ['c', 'celsius']:
            return value - 273.15
        elif from_unit in ['f', 'fahrenheit'] and to_unit in ['k', 'kelvin']:
            return ((value - 32) * 5/9) + 273.15
        elif from_unit in ['k', 'kelvin'] and to_unit in ['f', 'fahrenheit']:
            return ((value - 273.15) * 9/5) + 32
        
        # Length conversions (to meters)
        length_to_meters = {
            'mm': 0.001, 'millimeter': 0.001, 'millimeters': 0.001,
            'cm': 0.01, 'centimeter': 0.01, 'centimeters': 0.01,
            'm': 1.0, 'meter': 1.0, 'meters': 1.0,
            'km': 1000.0, 'kilometer': 1000.0, 'kilometers': 1000.0,
            'in': 0.0254, 'inch': 0.0254, 'inches': 0.0254,
            'ft': 0.3048, 'foot': 0.3048, 'feet': 0.3048,
            'yd': 0.9144, 'yard': 0.9144, 'yards': 0.9144,
            'mi': 1609.34, 'mile': 1609.34, 'miles': 1609.34
        }
        
        if from_unit in length_to_meters and to_unit in length_to_meters:
            meters = value * length_to_meters[from_unit]
            return meters / length_to_meters[to_unit]
        
        # Weight conversions (to grams)
        weight_to_grams = {
            'mg': 0.001, 'milligram': 0.001, 'milligrams': 0.001,
            'g': 1.0, 'gram': 1.0, 'grams': 1.0,
            'kg': 1000.0, 'kilogram': 1000.0, 'kilograms': 1000.0,
            'oz': 28.3495, 'ounce': 28.3495, 'ounces': 28.3495,
            'lb': 453.592, 'pound': 453.592, 'pounds': 453.592
        }
        
        if from_unit in weight_to_grams and to_unit in weight_to_grams:
            grams = value * weight_to_grams[from_unit]
            return grams / weight_to_grams[to_unit]
        
        # Volume conversions (to liters)
        volume_to_liters = {
            'ml': 0.001, 'milliliter': 0.001, 'milliliters': 0.001,
            'l': 1.0, 'liter': 1.0, 'liters': 1.0,
            'gal': 3.78541, 'gallon': 3.78541, 'gallons': 3.78541,
            'qt': 0.946353, 'quart': 0.946353, 'quarts': 0.946353,
            'pt': 0.473176, 'pint': 0.473176, 'pints': 0.473176,
            'cup': 0.236588, 'cups': 0.236588,
            'fl_oz': 0.0295735, 'fluid_ounce': 0.0295735
        }
        
        if from_unit in volume_to_liters and to_unit in volume_to_liters:
            liters = value * volume_to_liters[from_unit]
            return liters / volume_to_liters[to_unit]
        
        # If no conversion found
        raise ValueError(f"Cannot convert from {from_unit} to {to_unit}")
    
    async def cleanup(self) -> bool:
        """Cleanup calculator plugin resources"""
        self.logger.info("Calculator plugin cleaned up")
        return True