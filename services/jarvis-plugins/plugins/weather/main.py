"""
Weather Plugin for Jarvis v2.0
Provides weather information using OpenWeatherMap API
"""

import asyncio
import aiohttp
from typing import List
from plugin_system import CommandPlugin, PluginMetadata, PluginType, PluginPriority, PluginContext, PluginResponse, PluginConfig

class WeatherPlugin(CommandPlugin):
    """Weather information plugin"""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="weather",
            version="1.0.0",
            description="Get weather information for any city",
            author="Jarvis Team",
            plugin_type=PluginType.COMMAND,
            priority=PluginPriority.NORMAL,
            dependencies=["aiohttp"],
            permissions=["network"],
            config_schema={
                "api_key": {
                    "type": "string",
                    "description": "OpenWeatherMap API key",
                    "required": True
                },
                "default_city": {
                    "type": "string", 
                    "description": "Default city for weather",
                    "default": "Bangkok"
                },
                "units": {
                    "type": "string",
                    "description": "Temperature units (metric/imperial)",
                    "default": "metric"
                }
            },
            tags=["weather", "information", "api"]
        )
    
    async def initialize(self, config: PluginConfig) -> bool:
        """Initialize weather plugin"""
        try:
            self.config = config
            self.api_key = self.get_config_value("api_key")
            self.default_city = self.get_config_value("default_city", "Bangkok")
            self.units = self.get_config_value("units", "metric")
            
            if not self.api_key:
                self.logger.error("OpenWeatherMap API key not configured")
                return False
            
            # Test API connection
            test_result = await self.get_weather_data(self.default_city)
            if not test_result:
                self.logger.warning("Could not connect to weather API")
            
            self.logger.info("Weather plugin initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize weather plugin: {e}")
            return False
    
    def get_commands(self) -> List[str]:
        return ["weather", "w", "forecast"]
    
    async def handle_command(self, command: str, args: List[str], context: PluginContext) -> PluginResponse:
        """Handle weather commands"""
        try:
            if command in ["weather", "w"]:
                return await self.handle_weather_command(args, context)
            elif command == "forecast":
                return await self.handle_forecast_command(args, context)
            
            return PluginResponse(
                success=False,
                error=f"Unknown weather command: {command}"
            )
            
        except Exception as e:
            self.logger.error(f"Error in weather command: {e}")
            return PluginResponse(
                success=False,
                error=f"Weather service error: {str(e)}"
            )
    
    async def handle_weather_command(self, args: List[str], context: PluginContext) -> PluginResponse:
        """Handle current weather request"""
        city = " ".join(args) if args else self.default_city
        
        weather_data = await self.get_weather_data(city)
        if not weather_data:
            return PluginResponse(
                success=False,
                error=f"Could not get weather data for {city}"
            )
        
        # Format weather response
        temp = weather_data["main"]["temp"]
        feels_like = weather_data["main"]["feels_like"]
        humidity = weather_data["main"]["humidity"]
        description = weather_data["weather"][0]["description"]
        city_name = weather_data["name"]
        country = weather_data["sys"]["country"]
        
        unit_symbol = "°C" if self.units == "metric" else "°F"
        
        response_text = f"""🌤️ **Weather in {city_name}, {country}**
        
**Current:** {temp}{unit_symbol} (feels like {feels_like}{unit_symbol})
**Condition:** {description.title()}
**Humidity:** {humidity}%

*Powered by OpenWeatherMap*"""
        
        return PluginResponse(
            success=True,
            result=response_text,
            data={
                "city": city_name,
                "country": country,
                "temperature": temp,
                "feels_like": feels_like,
                "humidity": humidity,
                "description": description,
                "units": self.units
            },
            should_continue=False  # Weather response should be final
        )
    
    async def handle_forecast_command(self, args: List[str], context: PluginContext) -> PluginResponse:
        """Handle weather forecast request"""
        city = " ".join(args) if args else self.default_city
        
        forecast_data = await self.get_forecast_data(city)
        if not forecast_data:
            return PluginResponse(
                success=False,
                error=f"Could not get forecast data for {city}"
            )
        
        # Format forecast response (next 3 days)
        city_name = forecast_data["city"]["name"]
        country = forecast_data["city"]["country"]
        
        unit_symbol = "°C" if self.units == "metric" else "°F"
        
        forecast_text = f"📅 **3-Day Forecast for {city_name}, {country}**\\n\\n"
        
        # Group by day and get daily forecasts
        daily_forecasts = {}
        for item in forecast_data["list"][:8]:  # Next 24 hours (3-hour intervals)
            date = item["dt_txt"].split()[0]
            if date not in daily_forecasts:
                daily_forecasts[date] = []
            daily_forecasts[date].append(item)
        
        for date, forecasts in list(daily_forecasts.items())[:3]:
            day_temps = [f["main"]["temp"] for f in forecasts]
            descriptions = [f["weather"][0]["description"] for f in forecasts]
            
            min_temp = min(day_temps)
            max_temp = max(day_temps)
            main_desc = max(set(descriptions), key=descriptions.count)
            
            forecast_text += f"**{date}:** {min_temp}-{max_temp}{unit_symbol}, {main_desc.title()}\\n"
        
        forecast_text += "\\n*Powered by OpenWeatherMap*"
        
        return PluginResponse(
            success=True,
            result=forecast_text,
            data={
                "city": city_name,
                "country": country,
                "forecast": daily_forecasts,
                "units": self.units
            },
            should_continue=False
        )
    
    async def get_weather_data(self, city: str) -> dict:
        """Get current weather data from API"""
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather"
            params = {
                "q": city,
                "appid": self.api_key,
                "units": self.units
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        self.logger.error(f"Weather API error: {response.status}")
                        return None
                        
        except Exception as e:
            self.logger.error(f"Error fetching weather data: {e}")
            return None
    
    async def get_forecast_data(self, city: str) -> dict:
        """Get forecast data from API"""
        try:
            url = f"http://api.openweathermap.org/data/2.5/forecast"
            params = {
                "q": city,
                "appid": self.api_key,
                "units": self.units
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        self.logger.error(f"Forecast API error: {response.status}")
                        return None
                        
        except Exception as e:
            self.logger.error(f"Error fetching forecast data: {e}")
            return None
    
    async def cleanup(self) -> bool:
        """Cleanup weather plugin resources"""
        self.logger.info("Weather plugin cleaned up")
        return True