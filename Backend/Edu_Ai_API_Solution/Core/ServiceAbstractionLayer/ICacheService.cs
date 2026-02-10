using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceAbstractionLayer
{
    public interface ICacheService
    {
        public Task<string?> GetAsync(string key);
        public  Task SetAsync(string key, object value, TimeSpan ttl);
    }

    
}
