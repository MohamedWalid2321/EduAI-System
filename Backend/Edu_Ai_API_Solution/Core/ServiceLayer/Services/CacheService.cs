using DomainLayer.Contracts;
using ServiceAbstractionLayer;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceLayer.Services
{
    public class CacheService : ICacheService
    {
        private readonly IRedisService _redis;

        public CacheService(IRedisService redis)
        {
            _redis = redis;
        }

        public Task<string?> GetAsync(string key)
            => _redis.GetKeyAsync(key);

        public Task SetAsync(string key, object value, TimeSpan ttl)
            => _redis.SetKeyAsync(key, value, ttl);
    }

}
