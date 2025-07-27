"""
Time Plugin for Jarvis v2.0
Provides time and date information for different timezones
"""

from datetime import datetime, timezone
from typing import List
import pytz
from plugin_system import CommandPlugin, PluginMetadata, PluginType, PluginPriority, PluginContext, PluginResponse, PluginConfig

class TimePlugin(CommandPlugin):
    """Time and date information plugin"""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="time",
            version="1.0.0",
            description="Get current time and date for any timezone",
            author="Jarvis Team",
            plugin_type=PluginType.COMMAND,
            priority=PluginPriority.NORMAL,
            dependencies=["pytz"],
            permissions=[],
            config_schema={
                "default_timezone": {
                    "type": "string",
                    "description": "Default timezone",
                    "default": "Asia/Bangkok"
                },
                "time_format": {
                    "type": "string",
                    "description": "Time format string",
                    "default": "%H:%M:%S"
                },
                "date_format": {
                    "type": "string", 
                    "description": "Date format string",
                    "default": "%Y-%m-%d"
                }
            },
            tags=["time", "date", "timezone", "utility"]
        )
    
    async def initialize(self, config: PluginConfig) -> bool:
        """Initialize time plugin"""
        try:
            self.config = config
            self.default_timezone = self.get_config_value("default_timezone", "Asia/Bangkok")
            self.time_format = self.get_config_value("time_format", "%H:%M:%S")
            self.date_format = self.get_config_value("date_format", "%Y-%m-%d")
            
            # Test timezone
            try:
                pytz.timezone(self.default_timezone)
            except pytz.UnknownTimeZoneError:
                self.logger.warning(f"Unknown timezone: {self.default_timezone}, using UTC")
                self.default_timezone = "UTC"
            
            self.logger.info("Time plugin initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize time plugin: {e}")
            return False
    
    def get_commands(self) -> List[str]:
        return ["time", "date", "datetime", "timezone", "tz"]
    
    async def handle_command(self, command: str, args: List[str], context: PluginContext) -> PluginResponse:
        """Handle time commands"""
        try:
            if command == "time":
                return await self.handle_time_command(args, context)
            elif command == "date":
                return await self.handle_date_command(args, context)
            elif command == "datetime":
                return await self.handle_datetime_command(args, context)
            elif command in ["timezone", "tz"]:
                return await self.handle_timezone_command(args, context)
            
            return PluginResponse(
                success=False,
                error=f"Unknown time command: {command}"
            )
            
        except Exception as e:
            self.logger.error(f"Error in time command: {e}")
            return PluginResponse(
                success=False,
                error=f"Time service error: {str(e)}"
            )
    
    async def handle_time_command(self, args: List[str], context: PluginContext) -> PluginResponse:
        """Handle time request"""
        timezone_name = " ".join(args) if args else self.default_timezone
        
        try:
            tz = pytz.timezone(timezone_name)
            current_time = datetime.now(tz)
            
            time_str = current_time.strftime(self.time_format)
            
            response_text = f"🕐 **Current Time**\\n\\n"
            response_text += f"**Time:** {time_str}\\n"
            response_text += f"**Timezone:** {timezone_name}\\n"
            response_text += f"**UTC Offset:** {current_time.strftime('%z')}"
            
            return PluginResponse(
                success=True,
                result=response_text,
                data={
                    "time": time_str,
                    "timezone": timezone_name,
                    "utc_offset": current_time.strftime('%z'),
                    "timestamp": current_time.timestamp()
                },
                should_continue=False
            )
            
        except pytz.UnknownTimeZoneError:
            return PluginResponse(
                success=False,
                error=f"Unknown timezone: {timezone_name}"
            )
    
    async def handle_date_command(self, args: List[str], context: PluginContext) -> PluginResponse:
        """Handle date request"""
        timezone_name = " ".join(args) if args else self.default_timezone
        
        try:
            tz = pytz.timezone(timezone_name)
            current_time = datetime.now(tz)
            
            date_str = current_time.strftime(self.date_format)
            day_name = current_time.strftime("%A")
            month_name = current_time.strftime("%B")
            
            response_text = f"📅 **Current Date**\\n\\n"
            response_text += f"**Date:** {date_str}\\n"
            response_text += f"**Day:** {day_name}\\n"
            response_text += f"**Month:** {month_name}\\n"
            response_text += f"**Timezone:** {timezone_name}"
            
            return PluginResponse(
                success=True,
                result=response_text,
                data={
                    "date": date_str,
                    "day": day_name,
                    "month": month_name,
                    "timezone": timezone_name,
                    "year": current_time.year,
                    "month_num": current_time.month,
                    "day_num": current_time.day
                },
                should_continue=False
            )
            
        except pytz.UnknownTimeZoneError:
            return PluginResponse(
                success=False,
                error=f"Unknown timezone: {timezone_name}"
            )
    
    async def handle_datetime_command(self, args: List[str], context: PluginContext) -> PluginResponse:
        """Handle datetime request"""
        timezone_name = " ".join(args) if args else self.default_timezone
        
        try:
            tz = pytz.timezone(timezone_name)
            current_time = datetime.now(tz)
            
            datetime_str = current_time.strftime(f"{self.date_format} {self.time_format}")
            day_name = current_time.strftime("%A")
            
            response_text = f"🕐📅 **Current Date & Time**\\n\\n"
            response_text += f"**DateTime:** {datetime_str}\\n"
            response_text += f"**Day:** {day_name}\\n"
            response_text += f"**Timezone:** {timezone_name}\\n"
            response_text += f"**UTC Offset:** {current_time.strftime('%z')}"
            
            return PluginResponse(
                success=True,
                result=response_text,
                data={
                    "datetime": datetime_str,
                    "day": day_name,
                    "timezone": timezone_name,
                    "utc_offset": current_time.strftime('%z'),
                    "timestamp": current_time.timestamp()
                },
                should_continue=False
            )
            
        except pytz.UnknownTimeZoneError:
            return PluginResponse(
                success=False,
                error=f"Unknown timezone: {timezone_name}"
            )
    
    async def handle_timezone_command(self, args: List[str], context: PluginContext) -> PluginResponse:
        """Handle timezone listing/info"""
        if not args:
            # List common timezones
            common_timezones = [
                "UTC",
                "US/Eastern", "US/Central", "US/Mountain", "US/Pacific",
                "Europe/London", "Europe/Paris", "Europe/Berlin",
                "Asia/Tokyo", "Asia/Bangkok", "Asia/Shanghai",
                "Australia/Sydney", "Australia/Melbourne"
            ]
            
            response_text = "🌍 **Common Timezones**\\n\\n"
            for tz in common_timezones:
                try:
                    tz_obj = pytz.timezone(tz)
                    current_time = datetime.now(tz_obj)
                    time_str = current_time.strftime(self.time_format)
                    response_text += f"**{tz}:** {time_str}\\n"
                except:
                    continue
            
            return PluginResponse(
                success=True,
                result=response_text,
                data={"timezones": common_timezones},
                should_continue=False
            )
        else:
            # Show specific timezone info
            timezone_name = " ".join(args)
            try:
                tz = pytz.timezone(timezone_name)
                current_time = datetime.now(tz)
                
                response_text = f"🌍 **Timezone Information**\\n\\n"
                response_text += f"**Timezone:** {timezone_name}\\n"
                response_text += f"**Current Time:** {current_time.strftime(f'{self.date_format} {self.time_format}')}\\n"
                response_text += f"**UTC Offset:** {current_time.strftime('%z')}\\n"
                response_text += f"**DST Active:** {'Yes' if current_time.dst() else 'No'}"
                
                return PluginResponse(
                    success=True,
                    result=response_text,
                    data={
                        "timezone": timezone_name,
                        "current_time": current_time.strftime(f'{self.date_format} {self.time_format}'),
                        "utc_offset": current_time.strftime('%z'),
                        "dst_active": bool(current_time.dst())
                    },
                    should_continue=False
                )
                
            except pytz.UnknownTimeZoneError:
                return PluginResponse(
                    success=False,
                    error=f"Unknown timezone: {timezone_name}"
                )
    
    async def cleanup(self) -> bool:
        """Cleanup time plugin resources"""
        self.logger.info("Time plugin cleaned up")
        return True