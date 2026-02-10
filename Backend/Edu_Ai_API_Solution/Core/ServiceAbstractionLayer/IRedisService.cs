using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace ServiceAbstractionLayer
{
    public interface IRedisService
    {
        public Task SetKeyAsync(string cacheKey, object cacheValue, TimeSpan? ttl = null);

        public Task<string?> GetKeyAsync(string key);
        
    }
}
